"""
Phase 3, step 1: CPU QLoRA fine-tuning on the Asus NUC (Core Ultra 9 185H, 96GB).

Two supported CPU stacks -- pick ONE and uncomment its deps in
requirements-phase3.txt:

  A) intel-extension-for-transformers (ITREX): first to ship CPU QLoRA (INT4/NF4).
  B) ipex-llm: Intel's actively-maintained LLM library; QLoRA on CPU + XPU.

The training loop below is standard PEFT/transformers; the ONLY CPU-specific bit
is the 4-bit loading path (marked). On CPU expect ~12-24h for a 3B model over a
small corpus. When you add a GPU later, switch bnb 4-bit + device_map="cuda" and
the same script runs an order of magnitude faster.

Run:
    python -m phase3_rag.finetune_qlora --config config/config.yaml \
        --train data/processed/train.jsonl --val data/processed/val.jsonl
"""
from __future__ import annotations
import argparse, json, logging, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.finetune")

PROMPT = ("### Instruction:\n{instruction}\n\n### Response:\n{output}")


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def build(cfg, train_path, val_path):
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              TrainingArguments, Trainer, DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model

    base = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token

    # -------- 4-bit CPU load (stack A: ITREX) --------
    # from intel_extension_for_transformers.transformers import AutoModelForCausalLM as ItrexAMC
    # from intel_extension_for_transformers.transformers import RtnConfig
    # model = ItrexAMC.from_pretrained(base, quantization_config=RtnConfig(bits=4, compute_dtype="int8"),
    #                                  use_neural_speed=False, trust_remote_code=True)
    #
    # -------- 4-bit CPU load (stack B: ipex-llm) --------
    # from ipex_llm.transformers import AutoModelForCausalLM as IpexAMC
    # model = IpexAMC.from_pretrained(base, load_in_low_bit="nf4", optimize_model=True,
    #                                 trust_remote_code=True)
    #
    # -------- portable fallback (fp32/bf16 CPU; slower, no special deps) --------
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True)

    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["rank"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def fmt(ex):
        text = PROMPT.format(instruction=ex["instruction"], output=ex["output"])
        out = tok(text, truncation=True, max_length=cfg["train"]["max_seq_len"],
                  padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    ds_train = Dataset.from_list(load_jsonl(train_path)).map(fmt, remove_columns=["instruction","input","output","meta"])
    ds_val = Dataset.from_list(load_jsonl(val_path)).map(fmt, remove_columns=["instruction","input","output","meta"]) if os.path.exists(val_path) else None

    t = cfg["train"]
    args = TrainingArguments(
        output_dir="artifacts/lora-adapter", num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"], gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"], logging_steps=10, save_strategy="epoch",
        bf16=True, use_cpu=True, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    model.save_pretrained("artifacts/lora-adapter")
    tok.save_pretrained("artifacts/lora-adapter")
    log.info("saved LoRA adapter -> artifacts/lora-adapter")
    log.info("NEXT: merge + export GGUF, then `ollama create` (see serve.py header)")


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))["phase3"]
    build(cfg, a.train, a.val)


if __name__ == "__main__":
    main()
