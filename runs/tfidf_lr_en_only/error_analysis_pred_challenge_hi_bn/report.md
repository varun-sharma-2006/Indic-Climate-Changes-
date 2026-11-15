# Error analysis: runs\tfidf_lr_en_only\pred_challenge_hi_bn.csv

```

macro-F1 0.3300 | acc 0.3750 | MCC 0.2351 | F1(fav,ag) 0.2712 | n=64
  Favour   P 0.778  R 0.241  F1 0.368  (n=29)
  Against  P 0.667  R 0.100  F1 0.174  (n=20)
  None     P 0.288  R 1.000  F1 0.448  (n=15)
```

## Confusion matrix
```
              pred:Favour  pred:Against  pred:None
gold:Favour             7             1         21
gold:Against            2             2         16
gold:None               0             0         15
```

## By `phenomenon`
```
     phenomenon  n  macro_f1  accuracy
        codemix  1    0.0000    0.0000
 codemix-script  2    0.2222    0.5000
       explicit  9    0.2444    0.2222
      hard-none  2    0.3333    1.0000
       implicit 13    0.0000    0.0000
       negation  8    0.1667    0.2500
     rhetorical  1    0.0000    0.0000
      romanized  5    0.3333    0.6000
        sarcasm  8    0.0952    0.1250
topical-neutral  5    0.3333    1.0000
           typo  2    0.0000    0.0000
      unrelated  8    0.3333    1.0000
```

## By `lang`
```
  lang  n  macro_f1  accuracy
    bn 22    0.1282    0.2273
bn-rom  6    0.8222    0.8333
    en  2    0.6667    1.0000
    hi 25    0.1889    0.2800
hi-rom  9    0.4444    0.5556
```

## By `detected_lang`
```
detected_lang  n  macro_f1  accuracy
           bn 22    0.1282    0.2273
       bn-rom  6    0.8222    0.8333
           en  2    0.6667    1.0000
           hi 25    0.1889    0.2800
       hi-rom  9    0.4444    0.5556
```

## By `len_bin`
```
len_bin  n  macro_f1  accuracy
    1-5  2    0.2222    0.5000
  11-20 26    0.1340    0.1538
   6-10 36    0.4778    0.5278
```

## By `has_negation`
```
has_negation  n  macro_f1  accuracy
       False 49    0.3934    0.4490
        True 15    0.0889    0.1333
```

## By `has_climate_term`
```
has_climate_term  n  macro_f1  accuracy
           False 21    0.2963    0.5238
            True 43    0.2947    0.3023
```

## By `has_denial_term`
```
has_denial_term  n  macro_f1  accuracy
          False 57    0.3319    0.4035
           True  7    0.0833    0.1429
```

## By `code_mixed`
```
code_mixed  n  macro_f1  accuracy
     False 45    0.1750    0.2667
      True 19    0.5631    0.6316
```

## Accuracy by confidence bin (reliability)
```
             mean  count
conf_bin                
(0.4, 0.5]  0.500      2
(0.5, 0.6]  0.400      5
(0.6, 0.7]  0.429      7
(0.7, 0.8]  0.304     46
(0.8, 0.9]  1.000      4
```

## Most confident errors (top 25)

| gold | pred | conf | text |
|---|---|---|---|
| Favour | None | 0.79 | বিশ্ব উস্নায়ন খুব বিপজ্জনক ভাই!!!! 😱😱 |
| Favour | None | 0.79 | ग्लोबल वार्मिग बहोत खतरनाक हे भाई!!!! 😱😱 |
| Favour | Against | 0.78 | Global warming এর জন্য আমরাই দায়ী, নিজেরা না বদলালে কিছু হবে না |
| Favour | None | 0.73 | ग्लोबल वार्मिंग सच में बहुत गंभीर समस्या है, अभी नहीं संभले तो बहुत देर हो जाएगी। |
| Against | None | 0.73 | जलवायु परिवर्तन कोई बड़ी समस्या नहीं है, असली समस्या तो बेरोज़गारी है। |
| Favour | None | 0.73 | हर साल गर्मी बढ़ती जा रही है, ये जलवायु परिवर्तन का ही असर है। सरकार को कुछ करना चाहिए। |
| Favour | None | 0.73 | पेड़ लगाओ, पानी बचाओ, वरना आने वाली पीढ़ी हमें माफ़ नहीं करेगी 🙏 |
| Against | None | 0.73 | ग्लोबल वार्मिंग सब झूठ है, ये बस विदेशी कंपनियों का प्रोपेगेंडा है। |
| Favour | None | 0.73 | চমৎকার, পুকুর বুজিয়ে ফ্ল্যাট বানাও, তারপর গরমে কাঁদো 👏 |
| Against | None | 0.73 | আগে দেশের বেকারত্ব দূর করুন, গ্লোবাল ওয়ার্মিং নিয়ে পরে ভাবা যাবে। |
| Favour | None | 0.73 | যারা বলে বিশ্ব উষ্ণায়ন বাস্তব নয়, তারা একদম ভুল। |
| Against | None | 0.73 | हमारे दादा जी के ज़माने में भी इतनी ही गर्मी थी, सब बकवास है। |
| Favour | None | 0.73 | প্রতি বছর গরম বাড়ছে, এটা জলবায়ু পরিবর্তনেরই ফল। |
| Favour | None | 0.73 | বিশ্ব উষ্ণায়ন সত্যিই একটা ভয়ঙ্কর সমস্যা, এখনই ব্যবস্থা নেওয়া দরকার। |
| Favour | None | 0.73 | ये मज़ाक नहीं है, जलवायु संकट सच में हमारे दरवाज़े पर है। |
| Favour | None | 0.73 | जो लोग कहते हैं कि ग्लोबल वार्मिंग असली नहीं है, वो बिल्कुल गलत हैं। |
| Against | None | 0.73 | গ্লোবাল ওয়ার্মিং সব মিথ্যা কথা, বিদেশিদের ষড়যন্ত্র। |
| Favour | None | 0.73 | সরকার কিছু করবে না, আমাদেরই পরিবেশ বাঁচাতে হবে। |
| Against | None | 0.73 | গরম তো আগেও পড়ত, এখন সবাই অযথা ভয় দেখাচ্ছে। |
| Against | None | 0.73 | জলবায়ু পরিবর্তন কোনো বড় সমস্যা নয়, আসল সমস্যা হলো দারিদ্র্য। |
| Favour | None | 0.73 | সুন্দরবন ডুবে যাচ্ছে, আর আমরা চুপ করে বসে আছি। |
| Favour | None | 0.73 | গাছ কাটা বন্ধ করুন, না হলে আমাদের ভবিষ্যৎ অন্ধকার। |
| Against | None | 0.73 | पहले देश की गरीबी दूर करो, ये ग्लोबल वार्मिंग वगैरह बाद में देखेंगे। |
| Against | None | 0.73 | हाँ हाँ, ग्लोबल वार्मिंग से तो दुनिया कल ही खत्म हो जाएगी 🙄😂 |
| Against | None | 0.73 | হ্যাঁ হ্যাঁ, গ্লোবাল ওয়ার্মিংয়ে কালকেই পৃথিবী শেষ হয়ে যাবে 🙄😂 |

## Error type counts
```
label    pred   
Against  Favour      2
         None       16
Favour   Against     1
         None       21
```