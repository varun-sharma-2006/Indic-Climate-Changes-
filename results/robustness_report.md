# Robustness report (data never used for training)

|                    | muril_tt   | muril_tt_rom   | muril_robust1   |
|:-------------------|:-----------|:---------------|:----------------|
| challenge F1 (64)  | 0.6576     | 0.6303         | 0.7232          |
| challenge acc      | 0.6719     | 0.6562         | 0.7344          |
| sarcasm acc        | 0/8        | 0/8            | 4/8             |
| romanized acc      | 3/5        | 3/5            | 3/5             |
| implicit acc       | 8/13       | 9/13           | 9/13            |
| negation acc       | 5/8        | 3/8            | 4/8             |
| romanised-lang acc | 9/15       | 10/15          | 11/15           |
| sanity (18)        | 16/18      | 16/18          | 17/18           |
| SemEval2016 bn F1  | 0.425      | 0.4191         | 0.4965          |
| SemEval2016 en F1  | 0.5569     | 0.5642         | 0.5803          |
| SemEval2016 hi F1  | 0.4577     | 0.4676         | 0.5176          |

## Sarcasm examples in the challenge set

| model         | text                                                                                | true    | pred    |
|:--------------|:------------------------------------------------------------------------------------|:--------|:--------|
| muril_tt      | हाँ हाँ, ग्लोबल वार्मिंग से तो दुनिया कल ही खत्म हो जाएगी 🙄😂                        | Against | Favour  |
| muril_tt      | वाह! इतनी ठंड में भी ग्लोबल वार्मिंग? कमाल है 😂                                     | Against | None    |
| muril_tt      | बढ़िया है, पेड़ काटते रहो, मॉल बनाते रहो, फिर AC में बैठ के रोना कि गर्मी बहुत है 👏 | Favour  | None    |
| muril_tt      | itni thand pad rahi hai aur ye log global warming bol rahe hain 🤣                   | Against | Favour  |
| muril_tt      | হ্যাঁ হ্যাঁ, গ্লোবাল ওয়ার্মিংয়ে কালকেই পৃথিবী শেষ হয়ে যাবে 🙄😂                    | Against | Favour  |
| muril_tt      | এত ঠান্ডার মধ্যেও গ্লোবাল ওয়ার্মিং? বাহ্‌ 😂                                        | Against | None    |
| muril_tt      | চমৎকার, পুকুর বুজিয়ে ফ্ল্যাট বানাও, তারপর গরমে কাঁদো 👏                             | Favour  | None    |
| muril_tt      | eto thanda porche ar era bolche global warming 🤣                                    | Against | None    |
| muril_tt_rom  | हाँ हाँ, ग्लोबल वार्मिंग से तो दुनिया कल ही खत्म हो जाएगी 🙄😂                        | Against | Favour  |
| muril_tt_rom  | वाह! इतनी ठंड में भी ग्लोबल वार्मिंग? कमाल है 😂                                     | Against | None    |
| muril_tt_rom  | बढ़िया है, पेड़ काटते रहो, मॉल बनाते रहो, फिर AC में बैठ के रोना कि गर्मी बहुत है 👏 | Favour  | None    |
| muril_tt_rom  | itni thand pad rahi hai aur ye log global warming bol rahe hain 🤣                   | Against | Favour  |
| muril_tt_rom  | হ্যাঁ হ্যাঁ, গ্লোবাল ওয়ার্মিংয়ে কালকেই পৃথিবী শেষ হয়ে যাবে 🙄😂                    | Against | Favour  |
| muril_tt_rom  | এত ঠান্ডার মধ্যেও গ্লোবাল ওয়ার্মিং? বাহ্‌ 😂                                        | Against | None    |
| muril_tt_rom  | চমৎকার, পুকুর বুজিয়ে ফ্ল্যাট বানাও, তারপর গরমে কাঁদো 👏                             | Favour  | None    |
| muril_tt_rom  | eto thanda porche ar era bolche global warming 🤣                                    | Against | None    |
| muril_robust1 | हाँ हाँ, ग्लोबल वार्मिंग से तो दुनिया कल ही खत्म हो जाएगी 🙄😂                        | Against | Against |
| muril_robust1 | वाह! इतनी ठंड में भी ग्लोबल वार्मिंग? कमाल है 😂                                     | Against | Against |
| muril_robust1 | बढ़िया है, पेड़ काटते रहो, मॉल बनाते रहो, फिर AC में बैठ के रोना कि गर्मी बहुत है 👏 | Favour  | None    |
| muril_robust1 | itni thand pad rahi hai aur ye log global warming bol rahe hain 🤣                   | Against | Favour  |
| muril_robust1 | হ্যাঁ হ্যাঁ, গ্লোবাল ওয়ার্মিংয়ে কালকেই পৃথিবী শেষ হয়ে যাবে 🙄😂                    | Against | Against |
| muril_robust1 | এত ঠান্ডার মধ্যেও গ্লোবাল ওয়ার্মিং? বাহ্‌ 😂                                        | Against | Against |
| muril_robust1 | চমৎকার, পুকুর বুজিয়ে ফ্ল্যাট বানাও, তারপর গরমে কাঁদো 👏                             | Favour  | None    |
| muril_robust1 | eto thanda porche ar era bolche global warming 🤣                                    | Against | None    |