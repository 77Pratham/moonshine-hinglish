import re, unicodedata

V_IND = {"अ":"a","आ":"aa","इ":"i","ई":"i","उ":"u","ऊ":"u","ऋ":"ri","ए":"e","ऐ":"ai",
         "ओ":"o","औ":"au","ऑ":"o","ऍ":"e"}
MATRA = {"ि":"i","ी":"i","ु":"u","ू":"u","ृ":"ri","े":"e","ै":"ai",
         "ो":"o","ौ":"au","ॉ":"o","ॅ":"e"}
CONS = {"क":"k","ख":"kh","ग":"g","घ":"gh","ङ":"n","च":"ch","छ":"chh","ज":"j","झ":"jh",
        "ञ":"n","ट":"t","ठ":"th","ड":"d","ढ":"dh","ण":"n","त":"t","थ":"th","द":"d",
        "ध":"dh","न":"n","प":"p","फ":"ph","ब":"b","भ":"bh","म":"m","य":"y","र":"r",
        "ल":"l","ळ":"l","व":"v","श":"sh","ष":"sh","स":"s","ह":"h"}
NUKTA = {"क":"q","ख":"kh","ग":"g","ज":"z","ड":"r","ढ":"rh","फ":"f"}
NUKTA_CH, VIRAMA, AA_MATRA = "\u093c", "\u094d", "ा"
NASAL = {"ं":"n","ँ":"n"}

LOANWORDS = {
    "में": "mein",
    "ऑफिस": "office", "ऑफ़िस": "office", "आफिस": "office", "ओफिस": "office",
    "फोन": "phone", "फ़ोन": "phone", "मोबाइल": "mobile", "एड्रेस": "address",
    "बैग": "bag", "आयरन": "iron", "टाइम": "time", "प्रॉब्लम": "problem",
    "प्रोब्लम": "problem", "प्रॉब्लेम": "problem", "स्कूल": "school", "कॉलेज": "college",
    "डॉक्टर": "doctor", "हॉस्पिटल": "hospital", "ट्रेन": "train", "टिकट": "ticket",
    "बैंक": "bank", "कार्ड": "card", "अकाउंट": "account", "पासवर्ड": "password",
    "ऑनलाइन": "online", "ऑफलाइन": "offline", "मीटिंग": "meeting", "ईमेल": "email",
    "इंटरनेट": "internet", "फोटो": "photo", "वीडियो": "video", "नंबर": "number",
    "क्लास": "class", "एग्जाम": "exam", "एक्जाम": "exam", "रिजल्ट": "result",
    "मार्केट": "market", "जॉब": "job", "सैलरी": "salary", "कंपनी": "company",
    "प्रोजेक्ट": "project", "कंप्यूटर": "computer", "लैपटॉप": "laptop", "ओके": "ok",
    "थैंक्यू": "thank you", "सॉरी": "sorry", "प्लीज": "please", "प्लीज़": "please",
    "मैडम": "madam", "हेलो": "hello", "हैलो": "hello", "ऐप": "app", "एप": "app",
    "सिम": "sim", "नेटवर्क": "network", "चार्ज": "charge", "बैटरी": "battery",
    "कैमरा": "camera", "टीवी": "tv", "बाइक": "bike", "कार": "car", "पुलिस": "police",
    "स्टेशन": "station", "रोड": "road", "होटल": "hotel", "मैच": "match",
    "क्रिकेट": "cricket", "गेम": "game", "लोन": "loan", "पेमेंट": "payment",
    "बिल": "bill", "सर्विस": "service", "सिस्टम": "system", "मशीन": "machine",
    "ऑप्शन": "option", "सब्जेक्ट": "subject", "टीचर": "teacher", "स्टूडेंट": "student",
    "फॉर्म": "form", "डिग्री": "degree", "रिपोर्ट": "report", "ऑर्डर": "order",
    "डिलीवरी": "delivery", "ऑफर": "offer", "डिस्काउंट": "discount",
    "लिए": "liye", "हुआ": "hua", "गया": "gaya", "गई": "gayi",
}
LOANWORDS = {unicodedata.normalize("NFC", k): v for k, v in LOANWORDS.items()}

def _word(w):
    units, i = [], 0        # unit = [consonant, vowel, inherent_schwa, is_matra_aa]
    while i < len(w):
        ch = w[i]
        if ch in CONS:
            c, i = CONS[ch], i + 1
            if i < len(w) and w[i] == NUKTA_CH:
                c, i = NUKTA.get(ch, c), i + 1
            if i < len(w) and w[i] == VIRAMA:
                units.append([c, "", False, False]); i += 1
            elif i < len(w) and w[i] == AA_MATRA:
                units.append([c, "aa", False, True]); i += 1
            elif i < len(w) and w[i] in MATRA:
                units.append([c, MATRA[w[i]], False, False]); i += 1
            else:
                units.append([c, "a", True, False])
        elif ch in V_IND:
            units.append(["", V_IND[ch], False, ch == "आ"]); i += 1
        elif ch in NASAL:
            units.append([NASAL[ch], "", False, False]); i += 1
        elif ch == "ः":
            units.append(["h", "", False, False]); i += 1
        else:
            i += 1
    n = len(units)
    if n > 1 and units[-1][2]:                      # final schwa: kamal, ghar
        units[-1][1], units[-1][2] = "", False
    for k in range(n - 2, 0, -1):                   # medial schwa: karna, chalte
        u = units[k]
        if u[2] and units[k-1][1] and units[k+1][1]:
            u[1] = ""
    for k, u in enumerate(units):                   # long aa only in first syllable of longer words
        if u[3] and not (k == 0 and n > 1):
            u[1] = "a"
    return "".join(c + v for c, v, _, _ in units)

def romanize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]*>", " ", text).replace("।", " ")
    out = []
    for tok in text.split():
        if tok in LOANWORDS:
            out.append(LOANWORDS[tok]); continue
        out.append(re.sub(r"[\u0900-\u097F]+", lambda m: _word(m.group()), tok))
    return norm(" ".join(out))

def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()

if __name__ == "__main__":
    for s in ["मैं घर जा रहा हूँ", "कल मीटिंग है", "हम काम करना चाहते हैं", "नहीं समझ आया",
              "स्थानीय कला रूप या शिल्प होसंगाबाद जिले के लिए एक"]:
        print(s, "->", romanize_text(s))