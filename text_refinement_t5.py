from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

class TextRefinementT5:
    def __init__(self, model_name="AventIQ-AI/T5-small-grammar-correction", device="cpu"):
        """
        Initializes the T5-Small module for text refinement.
        
        Args:
            model_name (str): Hugging Face model hub name (default: "AventIQ-AI/T5-small-grammar-correction").
                              This fine-tuned T5-small model is much better at GEC than the base model.
            device (str): "cpu".
        """
        print(f"[T5-Refinement] Loading model '{model_name}' on {device}...")
        self.device = device
        
        # Optimization: fast tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        
        # Optimization: Load as float32 for CPU (default) or consider dynamic quantization later.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # Apply dynamic quantization for CPU speedup (Standard PyTorch optimization)
        # DISABLE QUANTIZATION: Causes RuntimeError on Windows (qlinear_dynamic (ONEDNN))
        # if device == "cpu":
        #     try:
        #         self.model = torch.quantization.quantize_dynamic(
        #             self.model, {torch.nn.Linear}, dtype=torch.qint8
        #         )
        #         print("[T5-Refinement] Applied dynamic key-value quantization for CPU efficiency.")
        #     except Exception as e:
        #         print(f"[T5-Refinement] Warning: Could not quantize model: {e}")
                
        self.model.eval()

    def refine(self, text):
        """
        Refines the input text using T5.
        
        Args:
            text (str): Raw transcript to refine.
            
        Returns:
            str: Refined text.
        """
        if not text or len(text.strip()) == 0:
            return ""

        # T5-small fine-tuned for grammar correction usually takes 'grammar: ' prefix.
        # "AventIQ-AI/T5-small-grammar-correction" expects 'grammar: ' or similar.
        input_text = f"grammar: {text}"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        start_time = time.time()
        
        # Generation parameters optimized for speed
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=64,             # Short sentences expected
                num_beams=1,               # Greedy search (fastest)
                length_penalty=1.0
            )
            
        refined = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Fallback: if refinement returns empty (rare), return original
        if not refined or not refined.strip():
            return text

        inference_time = time.time() - start_time
        # print(f"[T5-Refinement] Refined in {inference_time:.3f}s")
        
        return refined
