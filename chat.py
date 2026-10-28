import os
import sys
import torch
import argparse
from threading import Thread
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig
)

# Terminal Colors for Premium CLI Look
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    print(f"\n{Colors.BLUE}{Colors.BOLD}" + "=" * 60)
    print(f"    🐾 VetGPT-2-7B Interactive Terminal Chat 🐾")
    print(f"    Base: Llama-2-7b-hf  |  Adapter: vetgpt-2-7b")
    print("=" * 60 + f"{Colors.ENDC}\n")

def check_hf_token():
    """Checks if Hugging Face token is available, since Llama-2 is a gated model."""
    if not os.environ.get("HF_TOKEN") and not os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
        print(f"{Colors.WARNING}[UYARI] Llama-2 gated (izin gerektiren) bir modeldir.")
        print(f"Eğer model indirme hatası alırsanız, lütfen Hugging Face token'ınızı ayarlayın:")
        print(f"Windows (CMD):  set HF_TOKEN=your_token_here")
        print(f"Windows (PowerShell): $env:HF_TOKEN=\"your_token_here\"")
        print(f"Ya da terminalde 'huggingface-cli login' komutunu çalıştırın.{Colors.ENDC}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="VetGPT Interactive Chat CLI")
    parser.add_argument(
        "--base-model", 
        type=str, 
        default="meta-llama/Llama-2-7b-hf", 
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter-model", 
        type=str, 
        default="ArcanaBT/vetgpt-2-7b", 
        help="PEFT adapter name or path"
    )
    parser.add_argument(
        "--load-in-8bit", 
        action="store_true", 
        help="Load the model in 8-bit quantization (requires bitsandbytes)"
    )
    parser.add_argument(
        "--load-in-4bit", 
        action="store_true", 
        help="Load the model in 4-bit quantization (requires bitsandbytes)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to run the model on (cuda or cpu)"
    )
    parser.add_argument(
        "--max-new-tokens", 
        type=int, 
        default=512, 
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Generation temperature"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9, 
        help="Generation top_p"
    )
    return parser.parse_args()

def main():
    print_header()
    check_hf_token()
    args = parse_args()

    # Model loading configurations
    kwargs = {}
    if args.load_in_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"
        print(f"{Colors.CYAN}[Bilgi] Model 8-bit quantization ile yükleniyor...{Colors.ENDC}")
    elif args.load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["device_map"] = "auto"
        print(f"{Colors.CYAN}[Bilgi] Model 4-bit quantization ile yükleniyor...{Colors.ENDC}")
    else:
        if args.device == "cuda":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            print(f"{Colors.CYAN}[Bilgi] Model CUDA (FP16) modunda yükleniyor...{Colors.ENDC}")
        else:
            kwargs["torch_dtype"] = torch.float32
            print(f"{Colors.CYAN}[Bilgi] Model CPU (FP32) modunda yükleniyor (Yavaş olabilir)...{Colors.ENDC}")

    try:
        # Load Tokenizer
        print(f"{Colors.CYAN}[1/3] Tokenizer yükleniyor: {args.base_model}...{Colors.ENDC}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        # Llama tokenizer might not have a pad token, set it to eos token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load Base Model
        print(f"{Colors.CYAN}[2/3] Ana model yükleniyor (bu işlem zaman alabilir): {args.base_model}...{Colors.ENDC}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            **kwargs
        )

        # Load PEFT Adapter
        print(f"{Colors.CYAN}[3/3] PEFT Adapter yükleniyor: {args.adapter_model}...{Colors.ENDC}")
        model = PeftModel.from_pretrained(base_model, args.adapter_model)
        
        # If not quantized and not device_map auto, move to device
        if not args.load_in_8bit and not args.load_in_4bit and args.device == "cpu":
            model = model.to(args.device)

        model.eval()
        print(f"\n{Colors.GREEN}{Colors.BOLD}[BAŞARILI] Model başarıyla yüklendi! Sohbet etmeye başlayabilirsiniz.{Colors.ENDC}")
        print(f"{Colors.WARNING}Çıkmak için 'exit' veya 'quit' yazın. Geçmişi temizlemek için 'clear' yazın.{Colors.ENDC}\n")

    except Exception as e:
        print(f"\n{Colors.FAIL}{Colors.BOLD}[HATA] Model yüklenirken bir sorun oluştu!{Colors.ENDC}")
        print(f"{Colors.FAIL}{str(e)}{Colors.ENDC}")
        print(f"\n{Colors.WARNING}Lütfen şunları kontrol edin:")
        print("1. Hugging Face hesabınızla Llama-2 modeline erişim izni aldınız mı?")
        print("2. HF_TOKEN çevre değişkenini doğru ayarladınız mı?")
        print("3. Gerekli kütüphaneleri (requirements.txt) yüklediniz mi?")
        print(f"4. Yeterli disk alanı ve VRAM/RAM kapasiteniz var mı?{Colors.ENDC}")
        sys.exit(1)

    # Initialize basic chat history
    chat_history = []
    
    # System prompt for veterinary expertise
    system_prompt = (
        "You are VetGPT, a professional veterinary medicine AI assistant. "
        "Provide accurate, evidence-based, species-appropriate guidance using calm, concise, and practical language. "
        "Always respond in the same language as the user. Ask focused follow-up questions when essential facts are missing, "
        "including species, breed, age, weight, sex/neuter status, symptoms, duration, severity, medical history, medications, "
        "diet, and recent exposure. Never invent clinical facts, measurements, diagnoses, prescriptions, doses, or test results. "
        "Clearly distinguish general educational information from a veterinarian's clinical diagnosis. "
        "For possible emergencies, give immediate safety steps and urgently advise contacting a veterinarian, emergency clinic, or poison-control service; "
        "use local emergency guidance when provided. Do not make or confirm a diagnosis, prescribe medication, recommend dangerous dosing, or replace an in-person examination. "
        "For animals that may be seriously ill, injured, struggling to breathe, collapsed, bleeding heavily, seizing, having repeated vomiting, "
        "or showing other time-critical signs, prioritize urgent professional care. "
        "Include cautious red flags, practical next steps, monitoring advice, and a clear recommendation to seek professional veterinary care when appropriate. "
        "Acknowledge uncertainty explicitly and explain important limitations without alarmist or vague language. "
        "Do not browse, claim real-world observations, or cite sources unless they are supplied in the conversation."
    )

    while True:
        try:
            user_input = input(f"{Colors.GREEN}{Colors.BOLD}Siz > {Colors.ENDC}").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print(f"\n{Colors.BLUE}Sohbet sonlandırıldı. Görüşmek üzere! 🐾{Colors.ENDC}\n")
                break

            if user_input.lower() == "clear":
                chat_history = []
                print(f"{Colors.CYAN}[Bilgi] Sohbet geçmişi temizlendi.{Colors.ENDC}\n")
                continue

            # Build prompt using the Llama-2 chat instruction format.
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

            # Keep at most the last three complete turns.
            recent_history = chat_history[-6:]
            for i in range(0, len(recent_history) - 1, 2):
                if recent_history[i][0] == "user" and recent_history[i + 1][0] == "assistant":
                    prompt += f"{recent_history[i][1]} [/INST] {recent_history[i + 1][1]} </s><s>[INST] "

            prompt += f"{user_input} [/INST]"

            # Tokenize input with an attention mask for stable generation.
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            # Setup streamer for token-by-token printing
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, clean_up_tokenization_spaces=True)

            # Generation configuration
            generation_config = GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generation_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                streamer=streamer,
                generation_config=generation_config
            )

            # Run generation in a separate thread so we can stream the output
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # Print response in real-time
            print(f"{Colors.BLUE}{Colors.BOLD}VetGPT > {Colors.ENDC}", end="", flush=True)
            
            full_response = ""
            for new_text in streamer:
                # Filter out system tags or formatting leaks if any
                print(new_text, end="", flush=True)
                full_response += new_text
            
            print("\n") # New line after generation completes

            # Avoid carrying formatting markers or transcript artifacts into history.
            full_response = full_response.replace("</s>", "").replace("<s>", "")

            # Save to history
            chat_history.append(("user", user_input))
            chat_history.append(("assistant", full_response.strip()))

        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}[Bilgi] İşlem iptal edildi.{Colors.ENDC}\n")
            continue
        except Exception as e:
            print(f"\n{Colors.FAIL}[HATA] Yanıt üretilirken bir hata oluştu: {str(e)}{Colors.ENDC}\n")

if __name__ == "__main__":
    main()