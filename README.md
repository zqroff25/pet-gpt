# VetGPT Terminal Chat

VetGPT, yerel terminal üzerinden ArcanaBT tarafından sağlanan veteriner odaklı yapay zekâ modelini kullanan bir sohbet uygulamasıdır. Temel olarak Meta Llama-2 7B modeli ile PEFT adapter modelini bir araya getirir.

## Gereksinimler

- Python 3.9 veya üzeri
- CUDA destekli ekran kartı önerilir, CPU ile çalışabilir
- Hugging Face hesabı
- Llama-2 modeli için Hugging Face erişim izni ve API anahtarı

## Kurulum

1. Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

2. Hugging Face API anahtarınızı ortam değişkeni olarak tanımlayın:

```powershell
$env:HF_TOKEN="your_huggingface_api_key"
```

## Çalıştırma

```bash
python chat.py
```

Sohbetten çıkmak için `exit` veya `quit`, geçmişi temizlemek için `clear` yazabilirsiniz.

## Önemli Not

Bu proje, Meta tarafından sunulan Llama-2 tabanlı model için Hugging Face üzerinden API anahtarı ve model erişim izni gerektirir. Anahtarınızı kaynak koduna eklemeyin veya paylaşmayın.
