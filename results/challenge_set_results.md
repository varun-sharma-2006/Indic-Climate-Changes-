# Challenge set results (64 hand-written comments)

Labels: {'Favour': 29, 'Against': 20, 'None': 15}  |  languages: {'hi': 25, 'bn': 22, 'hi-rom': 9, 'bn-rom': 6, 'en': 2}

## Overall

| model    |   macro F1 |   accuracy |   F1 hi |   F1 bn |
|:---------|-----------:|-----------:|--------:|--------:|
| xlmr_tt  |     0.6775 |     0.6875 |  0.7358 |  0.5575 |
| muril_tt |     0.6576 |     0.6719 |  0.6743 |  0.5944 |
| ensemble |     0.6576 |     0.6719 |  0.7059 |  0.5339 |

## Accuracy of `xlmr_tt` by phenomenon

| phenomenon      |   n |   accuracy |
|:----------------|----:|-----------:|
| codemix         |   1 |       1    |
| codemix-script  |   2 |       1    |
| explicit        |   9 |       1    |
| hard-none       |   2 |       1    |
| implicit        |  13 |       0.62 |
| negation        |   8 |       0.62 |
| rhetorical      |   1 |       1    |
| romanized       |   5 |       0.2  |
| sarcasm         |   8 |       0.38 |
| topical-neutral |   5 |       0.8  |
| typo            |   2 |       1    |
| unrelated       |   8 |       0.75 |

## Errors of `xlmr_tt`

- [hi, implicit] true **Favour**, predicted **None**: इस बार दिल्ली में 50 डिग्री पार... भगवान ही बचाए अब तो 😰
- [hi, negation] true **Favour**, predicted **Against**: जो लोग कहते हैं कि ग्लोबल वार्मिंग असली नहीं है, वो बिल्कुल गलत हैं।
- [hi, unrelated] true **None**, predicted **Favour**: मेरे चैनल को भी सब्सक्राइब करो 🙏🙏
- [hi, sarcasm] true **Against**, predicted **None**: वाह! इतनी ठंड में भी ग्लोबल वार्मिंग? कमाल है 😂
- [hi, sarcasm] true **Favour**, predicted **None**: बढ़िया है, पेड़ काटते रहो, मॉल बनाते रहो, फिर AC में बैठ के रोना कि गर्मी बहुत है 👏
- [hi, implicit] true **Against**, predicted **Favour**: पहले देश की गरीबी दूर करो, ये ग्लोबल वार्मिंग वगैरह बाद में देखेंगे।
- [hi-rom, romanized] true **Against**, predicted **None**: ye sab jhooth hai bhai, garmi to har saal padti hai
- [hi-rom, implicit] true **Favour**, predicted **None**: paani khatam ho raha hai, ped kat rahe hain, phir bhi log samajh nahi rahe
- [hi-rom, romanized] true **Favour**, predicted **None**: jalvayu parivartan pe dhyan do sarkaar, bachcho ka future khatre mein hai
- [bn, implicit] true **Against**, predicted **Favour**: গরম তো আগেও পড়ত, এখন সবাই অযথা ভয় দেখাচ্ছে।
- [bn, negation] true **Against**, predicted **Favour**: জলবায়ু পরিবর্তন কোনো বড় সমস্যা নয়, আসল সমস্যা হলো দারিদ্র্য।
- [bn, implicit] true **Against**, predicted **Favour**: আগে দেশের বেকারত্ব দূর করুন, গ্লোবাল ওয়ার্মিং নিয়ে পরে ভাবা যাবে।
- [bn, negation] true **Favour**, predicted **Against**: যারা বলে বিশ্ব উষ্ণায়ন বাস্তব নয়, তারা একদম ভুল।
- [bn, topical-neutral] true **None**, predicted **Favour**: আবহাওয়া দপ্তর বলেছে কাল থেকে কলকাতায় বৃষ্টি হবে।
- [bn, unrelated] true **None**, predicted **Favour**: এই চ্যানেলটা সাবস্ক্রাইব করুন সবাই 🙏
- [bn, sarcasm] true **Against**, predicted **None**: এত ঠান্ডার মধ্যেও গ্লোবাল ওয়ার্মিং? বাহ্‌ 😂
- [bn, sarcasm] true **Favour**, predicted **None**: চমৎকার, পুকুর বুজিয়ে ফ্ল্যাট বানাও, তারপর গরমে কাঁদো 👏
- [bn-rom, romanized] true **Favour**, predicted **Against**: global warming khub serious bepar, sobai mile kichu korte hobe
- [bn-rom, romanized] true **Against**, predicted **None**: eta sob bhua kotha, gorom to protibochor e pore
- [bn-rom, sarcasm] true **Against**, predicted **None**: eto thanda porche ar era bolche global warming 🤣