"""Full detailed transcript analysis of the Tribunal Ontario recording."""

import json
import re
from collections import defaultdict
from pathlib import Path

INPUT = Path(r"C:/Users/TheBeast/Videos/Radeon ReLive/unknown/35Marg.TribunalResolution_2026.09.14-09.53.json")

with open(INPUT) as f:
    data = json.load(f)

segs = data.get("segments", data) if isinstance(data, dict) else data

# ── helpers ──────────────────────────────────────────────────────────
def ts(s: float) -> str:
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m:02d}:{sec:02d}"

def fmt_dur(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

def pct(part: float, whole: float) -> str:
    return f"{part/whole*100:.1f}%"

# ── 1. Raw segment listing ─────────────────────────────────────────
print("=" * 100)
print("SECTION 1: COMPLETE CHRONOLOGICAL SEGMENT LIST")
print("=" * 100)

total_speech = 0
total_silence = 0
gaps = []

for i, seg in enumerate(segs):
    start = seg["start"]
    end = seg["end"]
    dur = end - start
    total_speech += dur
    text = seg["text"].strip()

    if i < len(segs) - 1:
        gap = segs[i + 1]["start"] - end
        if gap > 5:  # only show gaps > 5s
            total_silence += gap
            gaps.append((start, end, segs[i+1]["start"], gap))

    speaker_guess = ""
    # Heuristic speaker assignment based on context
    text_lower = text.lower()
    if any(w in text_lower for w in ["shahid", "gupta", "my name", "our tenants", "i had a question",
                                       "we are fine with her", "we would like to approach"]):
        speaker_guess = " [Landlord/Owner (likely Shahid Gupta)]"
    elif any(w in text_lower for w in ["pauline", "tribunal", "you'll have to go", "sheriff's department",
                                        "as far as i understand", "she's saying"]):
        speaker_guess = " [Tribunal Staff/Adjudicator (likely Pauline)]"
    elif any(w in text_lower for w in ["tara lee", "her staying"]):
        speaker_guess = " [Landlord discussing tenant]"
    elif text_lower.strip() in ("good morning", "yes", "no", "okay", "thank you", "hello", "good morning."):
        speaker_guess = " [Greeting/Response]"

    print(f"[{ts(start)}] {dur:5.1f}s{speaker_guess}")
    print(f"         {text}")
    print()

print(f"\nTotal speech duration: {fmt_dur(total_speech)}")
print(f"Total silence/gap duration: {fmt_dur(total_silence)}")
print(f"Total recording: {fmt_dur(data.get('duration', 3934))}")

# ── 2. Gap analysis ─────────────────────────────────────────────────
print("\n" + "=" * 100)
print("SECTION 2: SILENCE / GAP ANALYSIS (gaps > 30 seconds)")
print("=" * 100)

for i, (start, end, next_start, gap) in enumerate(gaps):
    if gap >= 30:
        print(f"  Gap {i+1}: [{ts(end)}] -> [{ts(next_start)}]  = {fmt_dur(gap)} "
              f"(before segment at {ts(next_start)})")
        # Show what comes after
        nxt_idx = None
        for j, s in enumerate(segs):
            if abs(s["start"] - next_start) < 0.5:
                nxt_idx = j
                break
        if nxt_idx is not None:
            print(f"         After gap: '{segs[nxt_idx]['text'][:80]}'")

# ── 3. Duration heatmap (by minute) ────────────────────────────────
print("\n" + "=" * 100)
print("SECTION 3: SPEECH HEATMAP (speech per minute)")
print("=" * 100)

minute_buckets = defaultdict(float)
for seg in segs:
    m = int(seg["start"] // 60)
    minute_buckets[m] += (seg["end"] - seg["start"])

total_min = int(data.get("duration", 3934) // 60)
max_dur = max(minute_buckets.values()) if minute_buckets else 1

for m in range(total_min):
    speech = minute_buckets.get(m, 0)
    bar_len = int(speech / max_dur * 40) if max_dur > 0 else 0
    bar = "#" * bar_len
    status = "SILENT" if speech < 10 else ("LIGHT" if speech < 40 else "ACTIVE")
    print(f"  {m:2d}:00  {speech:5.1f}s  {bar:40s} [{status}]")

# ── 4. Topic extraction ────────────────────────────────────────────
print("\n" + "=" * 100)
print("SECTION 4: TOPICAL BREAKDOWN")
print("=" * 100)

topics = {
    "GREETINGS / OPENING": [
        "good morning", "god bless", "hello", "thank you", "my name is"
    ],
    "PAYMENT PLAN TERMS": [
        "payment plan", "october 1st", "november 1st", "december 7th",
        "within 15 days", "18 636", "18,000", "amount due", "unpaid"
    ],
    "EVICTION PROCESS": [
        "sheriff", "eviction", "sheriff's department", "legal system",
        "bank", "get their salary", "get evicted"
    ],
    "Tribunal PROCEDURE": [
        "tribunal", "go back to", "waiting period", "two months",
        "you'll have to go"
    ],
    "TENANT DISCUSSION": [
        "tara lee", "tenant", "her staying", "payment continue",
        "two unit", "loud noises", "infighting", "complaining"
    ],
    "CLOSING": [
        "thank you very much", "thank you all", "god bless you all",
        "bye", "thank you everyone"
    ]
}

topic_segments = defaultdict(list)
for i, seg in enumerate(segs):
    text_lower = seg["text"].lower()
    for topic, keywords in topics.items():
        for kw in keywords:
            if kw in text_lower:
                topic_segments[topic].append((i, seg["start"], seg["text"][:100]))
                break

for topic, items in topic_segments.items():
    print(f"\n  [TOPIC] {topic}")
    print(f"  Segments: {len(items)}  |  Duration: {fmt_dur(sum(segs[i]['end']-segs[i]['start'] for i,_,_ in items))}")
    for idx, start, text in items:
        print(f"  [{ts(start)}] {text}")
    print()

# ── 5. Named entities / key facts ───────────────────────────────────
print("\n" + "=" * 100)
print("SECTION 5: KEY ENTITIES & FACTS")
print("=" * 100)

facts = []
full_text = " ".join(s["text"] for s in segs)

# Names
names_found = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', full_text)
name_counts = defaultdict(int)
for n in names_found:
    name_counts[n] += 1
# Filter to likely real names (not common words)
likely_names = {n: c for n, c in name_counts.items() if c >= 1 and n.lower() not in
                ("god bless", "sheriff's department", "tribunal of ontario", "tribunal ontario",
                 "ontario tribunal", "pauline",)}

print("\n  PEOPLE MENTIONED:")
people = {"Shahid Gupta": "Landlord/property owner (intro at 00:00)",
          "Pauline": "Tribunal staff/adjudicator",
          "Tara Lee": "Tenant (subject of eviction discussion)"}
for name, role in people.items():
    count = name_counts.get(name, 0)
    print(f"    * {name} ({count}x mention) - {role}")

# Financial
print("\n  FINANCIAL DETAILS:")
for seg in segs:
    text = seg["text"]
    if re.search(r'\$?\d[\d,]*\s*\d', text):
        # Extract numbers
        nums = re.findall(r'(\d[\d,]*)', text)
        for n in nums:
            clean = n.replace(",", "")
            if clean.isdigit():
                val = int(clean)
                if val >= 1000:
                    print(f"    * ${val:,} - found at [{ts(seg['start'])}]")

print("\n  DATES & DEADLINES:")
for seg in segs:
    text = seg["text"]
    if any(m in text.lower() for m in ["october", "november", "december", "day", "1st", "7th"]):
        print(f"    [{ts(seg['start'])}] {text[:100]}")

# Property details
print("\n  PROPERTY / TENANCY DETAILS:")
for seg in segs:
    text = seg["text"]
    if any(kw in text.lower() for kw in ["two unit", "rental", "tenant", "complaining", "loud noises"]):
        print(f"    [{ts(seg['start'])}] {text[:100]}")

# Legal procedure
print("\n  LEGAL PROCEDURE DISCUSSED:")
for seg in segs:
    text = seg["text"]
    if any(kw in text.lower() for kw in ["tribunal", "sheriff", "eviction", "waiting period", "legal system", "bank"]):
        print(f"    [{ts(seg['start'])}] {text[:100]}")

# ── 6. Transcription quality ───────────────────────────────────────
print("\n" + "=" * 100)
print("SECTION 6: TRANSCRIPTION QUALITY NOTES")
print("=" * 100)

issues = []

# Check for short segments that may be noise
short = [s for s in segs if (s["end"] - s["start"]) < 1.0]
print(f"\n  Short segments (<1s): {len(short)} (may be greetings, fillers)")
for s in short[:10]:
    print(f"    [{ts(s['start'])}] {s['text'][:60]}")

# Check for repetitions (hallucination indicator)
repeated = []
for i in range(len(segs)-1):
    t1 = segs[i]["text"].strip().lower().replace(".", "").replace(",", "")
    t2 = segs[i+1]["text"].strip().lower().replace(".", "").replace(",", "")
    if t1 == t2 and len(t1) > 5:
        repeated.append((i, t1))
if repeated:
    print(f"\n  Repeated segments: {len(repeated)} (possible transcription artifacts)")
    for idx, text in repeated[:5]:
        print(f"    Seg {idx}: '{text}'")

# Check for potential misrecognitions
print(f"\n  Potential issues to review:")
for seg in segs:
    text = seg["text"].strip()
    # Check for likely misheard names
    if "shahid" in text.lower() and "shraddha" in text.lower():
        print(f"    [{ts(seg['start'])}] Name conflict: 'Shahid' vs 'Shraddha' — verify against video")
    if "annouce" in text.lower():
        print(f"    [{ts(seg['start'])}] Possible misspelling: 'annouce' -> 'announce'")
    if "salary" in text.lower() and "legal system" in text.lower():
        print(f"    [{ts(seg['start'])}] 'salary' may be misheard - context suggests 'writ' or similar legal term")

# Check for long segments that may be speaker changes missed
long_segs = [s for s in segs if (s["end"] - s["start"]) > 30]
if long_segs:
    print(f"\n  Long segments (>30s — may contain multiple speakers):")
    for s in long_segs:
        print(f"    [{ts(s['start'])}] {fmt_dur(s['end']-s['start'])}: {s['text'][:100]}")

# ── 7. Segment-by-segment speaker attribution ──────────────────────
print("\n" + "=" * 100)
print("SECTION 7: DETAILED SEGMENT-BY-SEGMENT ANALYSIS")
print("=" * 100)

for i, seg in enumerate(segs):
    start = seg["start"]
    end = seg["end"]
    dur = end - start
    text = seg["text"].strip()

    # Speaker inference
    text_lower = text.lower()
    if i == 0 or "my name is" in text_lower or "shahid" in text_lower:
        spk = "A - Landlord (Shahid Gupta)"
    elif "pauline" in text_lower or "tribunal" in text_lower or "you'll have to" in text_lower or \
         "sheriff's department" in text_lower or "as far as i understand" in text_lower:
        spk = "B - Tribunal Staff (Pauline)"
    elif "tara lee" in text_lower or "her staying" in text_lower or "we are fine" in text_lower or \
         "we would like" in text_lower or "our tenants" in text_lower:
        spk = "A - Landlord (discussing tenant)"
    elif any(w in text_lower for w in ["good morning", "yes.", "no,", "hello", "thank you very"]):
        spk = "A/B - Shared/Greeting"
    elif "god bless" in text_lower:
        spk = "A/B - Closing blessing"
    elif "thank you" in text_lower and i > 60:
        spk = "A/B - Closing thanks"
    else:
        spk = "? - Unclear"

    print(f"\n  SEG {i:3d} | [{ts(start)} -> {ts(end)}] ({dur:5.1f}s)")
    print(f"         Speaker: {spk}")
    print(f"         Text: {text}")
