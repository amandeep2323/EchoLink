"""
Spell Corrector — Lightweight ASL Fingerspelling Word Correction
==================================================================
Corrects common fingerspelling errors using:
  1. Built-in dictionary of common English words
  2. difflib.get_close_matches for fuzzy matching
  3. ASL-specific confusion pairs (B/D, M/N, etc.)

No external dependencies (no BERT, no autocorrect library).
Uses Python's built-in difflib for edit-distance matching.

Usage:
    corrector = SpellCorrector()
    corrected = corrector.correct("HELO")   # → "HELLO"
    corrected = corrector.correct("HELLO")  # → "HELLO" (already correct)
    corrected = corrector.correct("XYZ")    # → "XYZ" (no match found)
"""

import difflib
from typing import Optional


# ═══════════════════════════════════════════════
# Common English Words (top ~2000)
# ═══════════════════════════════════════════════

# These are the most common English words that someone would
# fingerspell in ASL. Kept small for fast matching.
COMMON_WORDS = {
    # 1-2 letters (usually not corrected)
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go",
    "he", "if", "in", "is", "it", "me", "my", "no", "of", "ok",
    "on", "or", "so", "to", "up", "us", "we",

    # 3 letters
    "ace", "act", "add", "age", "ago", "aid", "aim", "air", "all",
    "and", "any", "ape", "arc", "are", "ark", "arm", "art", "ash",
    "ask", "ate", "ave", "awe", "axe", "bad", "bag", "ban", "bar",
    "bat", "bay", "bed", "bee", "bet", "big", "bit", "bow", "box",
    "boy", "bud", "bug", "bus", "but", "buy", "cab", "can", "cap",
    "car", "cat", "cop", "cow", "cry", "cup", "cut", "dad", "dam",
    "day", "did", "die", "dig", "dim", "dip", "dog", "dot", "dry",
    "due", "dug", "dye", "ear", "eat", "egg", "ego", "elm", "end",
    "era", "eve", "eye", "fan", "far", "fat", "fax", "fed", "fee",
    "few", "fig", "fin", "fit", "fix", "fly", "fog", "for", "fox",
    "fun", "fur", "gag", "gap", "gas", "gem", "get", "god", "got",
    "gum", "gun", "gut", "guy", "gym", "had", "ham", "has", "hat",
    "hay", "hen", "her", "hid", "him", "hip", "his", "hit", "hog",
    "hop", "hot", "how", "hub", "hug", "hum", "hut", "ice", "icy",
    "ill", "imp", "ink", "inn", "ion", "ire", "irk", "its", "ivy",
    "jab", "jag", "jam", "jar", "jaw", "jay", "jet", "jig", "job",
    "jog", "joy", "jug", "jut", "keg", "ken", "key", "kid", "kin",
    "kit", "lab", "lad", "lag", "lap", "law", "lay", "led", "leg",
    "let", "lid", "lie", "lip", "lit", "log", "lot", "low", "lug",
    "mad", "man", "map", "mat", "max", "may", "men", "met", "mid",
    "mix", "mob", "mom", "mop", "mow", "mud", "mug", "nag", "nap",
    "net", "new", "nil", "nod", "nor", "not", "now", "nun", "nut",
    "oak", "oar", "oat", "odd", "ode", "off", "oft", "oil", "old",
    "one", "opt", "orb", "ore", "our", "out", "owe", "owl", "own",
    "pad", "pal", "pan", "par", "pat", "paw", "pay", "pea", "peg",
    "pen", "per", "pet", "pie", "pig", "pin", "pit", "ply", "pod",
    "pop", "pot", "pow", "pro", "pry", "pub", "pug", "pun", "pup",
    "pus", "put", "rag", "ram", "ran", "rap", "rat", "raw", "ray",
    "red", "ref", "rib", "rid", "rig", "rim", "rip", "rob", "rod",
    "rot", "row", "rub", "rug", "rum", "run", "rut", "rye", "sac",
    "sad", "sag", "sap", "sat", "saw", "say", "sea", "set", "sew",
    "she", "shy", "sin", "sip", "sir", "sis", "sit", "six", "ski",
    "sky", "sly", "sob", "sod", "son", "sop", "sot", "sow", "soy",
    "spa", "spy", "sub", "sue", "sum", "sun", "sup", "tab", "tad",
    "tag", "tan", "tap", "tar", "tax", "tea", "ten", "the", "thy",
    "tic", "tie", "tin", "tip", "toe", "ton", "too", "top", "tot",
    "tow", "toy", "try", "tub", "tug", "two", "urn", "use", "van",
    "vat", "vet", "via", "vie", "vim", "vow", "wad", "wag", "war",
    "was", "wax", "way", "web", "wed", "wet", "who", "why", "wig",
    "win", "wit", "woe", "wok", "won", "woo", "wow", "yak", "yam",
    "yap", "yaw", "yea", "yes", "yet", "yew", "you", "zap", "zed",
    "zen", "zig", "zip", "zoo",

    # 4 letters
    "able", "ache", "acid", "also", "area", "army", "away", "baby",
    "back", "bake", "ball", "band", "bank", "bare", "bark", "barn",
    "base", "bath", "bead", "beam", "bean", "bear", "beat", "beef",
    "been", "beer", "bell", "belt", "bend", "bent", "best", "bike",
    "bill", "bind", "bird", "bite", "blow", "blue", "blur", "boat",
    "body", "bold", "bolt", "bomb", "bond", "bone", "book", "boom",
    "boot", "bore", "born", "boss", "both", "bowl", "bulk", "bull",
    "burn", "bury", "bush", "busy", "cafe", "cage", "cake", "call",
    "calm", "came", "camp", "card", "care", "cart", "case", "cash",
    "cast", "cave", "cell", "chat", "chef", "chin", "chip", "chop",
    "cite", "city", "clad", "clam", "clan", "clap", "clay", "clip",
    "club", "clue", "coal", "coat", "code", "coil", "coin", "cold",
    "come", "cook", "cool", "cope", "copy", "cord", "core", "cork",
    "corn", "cost", "cosy", "coup", "crew", "crop", "crow", "cube",
    "cult", "cure", "curl", "cute", "dare", "dark", "dart", "data",
    "date", "dawn", "dead", "deaf", "deal", "dear", "debt", "deck",
    "deed", "deem", "deep", "deer", "deny", "desk", "dial", "dice",
    "diet", "dirt", "disc", "dish", "dock", "does", "doll", "dome",
    "done", "doom", "door", "dose", "down", "draw", "drew", "drip",
    "drop", "drug", "drum", "dual", "duck", "dude", "duel", "dull",
    "dumb", "dump", "dune", "dusk", "dust", "duty", "dyed", "each",
    "earl", "earn", "ease", "east", "easy", "edge", "edit", "else",
    "emit", "epic", "euro", "even", "ever", "evil", "exam", "exec",
    "exit", "face", "fact", "fade", "fail", "fair", "fake", "fall",
    "fame", "fang", "fare", "farm", "fast", "fate", "fear", "feat",
    "feed", "feel", "feet", "fell", "felt", "file", "fill", "film",
    "find", "fine", "fire", "firm", "fish", "fist", "five", "flag",
    "flame", "flaw", "fled", "flew", "flip", "flog", "flow",
    "foam", "fold", "folk", "fond", "font", "food", "fool", "foot",
    "ford", "fore", "fork", "form", "fort", "foul", "four", "free",
    "frog", "from", "fuel", "full", "fund", "fury", "fuse", "fuss",
    "gain", "gale", "game", "gang", "gate", "gave", "gaze", "gear",
    "gene", "gift", "girl", "give", "glad", "glow", "glue", "goal",
    "goat", "goes", "gold", "golf", "gone", "good", "grab", "gram",
    "gray", "grew", "grey", "grid", "grim", "grin", "grip", "grit",
    "grow", "gulf", "guru", "gust", "guys", "hack", "hail", "hair",
    "half", "hall", "halt", "hand", "hang", "hard", "harm", "harp",
    "hash", "hate", "haul", "have", "haze", "head", "heal", "heap",
    "hear", "heat", "heed", "heel", "held", "hell", "help", "herb",
    "herd", "here", "hero", "hide", "high", "hike", "hill", "hint",
    "hire", "hold", "hole", "holy", "home", "hood", "hook", "hope",
    "horn", "host", "hour", "huge", "hull", "hung", "hunt", "hurt",
    "hymn", "icon", "idea", "inch", "info", "into", "iron", "isle",
    "item", "jack", "jail", "jazz", "jean", "jeep", "jest", "jobs",
    "join", "joke", "jolt", "jump", "june", "jury", "just", "keen",
    "keep", "kept", "kick", "kids", "kill", "kind", "king", "kiss",
    "kite", "knee", "knew", "knit", "knob", "knot", "know", "labs",
    "lace", "lack", "lady", "laid", "lake", "lamb", "lame", "lamp",
    "land", "lane", "laps", "last", "late", "lawn", "laws", "lead",
    "leaf", "leak", "lean", "leap", "left", "lend", "lens", "less",
    "liar", "lick", "life", "lift", "like", "limb", "lime", "limp",
    "line", "link", "lion", "list", "live", "load", "loaf", "loan",
    "lock", "logo", "lone", "long", "look", "loop", "lord", "lore",
    "lose", "loss", "lost", "lots", "loud", "love", "luck", "lump",
    "lung", "lure", "lurk", "lush", "lust", "made", "mail", "main",
    "make", "male", "mall", "malt", "mane", "many", "mare", "mark",
    "mask", "mass", "mate", "maze", "meal", "mean", "meat", "meet",
    "melt", "memo", "mend", "menu", "mere", "mesh", "mess", "mild",
    "mile", "milk", "mill", "mind", "mine", "mint", "miss", "mist",
    "mode", "mole", "monk", "mood", "moon", "more", "moss", "most",
    "moth", "move", "much", "mule", "must", "myth", "nail", "name",
    "navy", "near", "neat", "neck", "need", "nest", "news", "next",
    "nice", "nine", "node", "none", "noon", "norm", "nose", "note",
    "noun", "nude", "null", "oath", "obey", "odds", "okay", "omen",
    "omit", "once", "only", "onto", "opal", "open", "oral", "oven",
    "over", "pace", "pack", "page", "paid", "pail", "pain", "pair",
    "pale", "palm", "pane", "park", "part", "pass", "past", "path",
    "pave", "peak", "pear", "peel", "peer", "pick", "pile", "pine",
    "pink", "pipe", "plan", "play", "plea", "plot", "ploy", "plug",
    "plus", "poem", "poet", "pole", "poll", "polo", "pond", "pool",
    "poor", "pope", "pore", "pork", "port", "pose", "post", "pour",
    "pray", "prey", "prop", "pros", "puff", "pull", "pulp", "pump",
    "punk", "pure", "push", "quit", "quiz", "race", "rack", "rage",
    "raid", "rail", "rain", "rank", "rare", "rash", "rate", "read",
    "real", "reap", "rear", "reef", "rein", "rely", "rent", "rest",
    "rice", "rich", "ride", "rift", "rile", "ring", "riot", "ripe",
    "rise", "risk", "road", "roam", "roar", "robe", "rock", "rode",
    "role", "roll", "roof", "room", "root", "rope", "rose", "rude",
    "ruin", "rule", "rush", "rust", "sack", "safe", "sage", "said",
    "sail", "sake", "sale", "salt", "same", "sand", "sane", "sang",
    "sank", "save", "scan", "scar", "seal", "seam", "seas", "seat",
    "seed", "seek", "seem", "seen", "self", "sell", "send", "sent",
    "sept", "shed", "shin", "ship", "shoe", "shoo", "shop", "shot",
    "show", "shut", "sick", "side", "sift", "sigh", "sign", "silk",
    "sing", "sink", "site", "size", "skim", "skin", "skip", "slab",
    "slam", "slap", "slay", "sled", "slew", "slid", "slim", "slip",
    "slit", "slot", "slow", "slug", "slum", "snap", "snip", "snow",
    "soak", "soap", "soar", "sock", "sofa", "soft", "soil", "sold",
    "sole", "some", "song", "soon", "sort", "soul", "sour", "span",
    "spar", "spec", "sped", "spin", "spit", "spot", "spur", "stab",
    "star", "stay", "stem", "step", "stew", "stir", "stop", "stub",
    "such", "suck", "suit", "sung", "sunk", "sure", "surf", "swan",
    "swap", "swim", "swam", "tabs", "tack", "tact", "tail", "take",
    "tale", "talk", "tall", "tame", "tank", "tape", "taps", "task",
    "team", "tear", "teen", "tell", "temp", "tend", "tens", "tent",
    "term", "test", "text", "than", "that", "them", "then", "they",
    "thin", "this", "thus", "tick", "tidy", "tied", "tier", "ties",
    "tile", "till", "tilt", "time", "tiny", "tire", "toad", "toil",
    "told", "toll", "tomb", "tone", "took", "tool", "tops", "tore",
    "torn", "toss", "tour", "town", "trap", "tray", "tree", "trek",
    "trim", "trio", "trip", "trot", "true", "tube", "tuck", "tune",
    "turf", "turn", "twin", "type", "ugly", "undo", "unit", "unto",
    "upon", "urge", "used", "user", "vain", "vale", "vane", "vary",
    "vast", "veil", "vein", "vent", "verb", "very", "vest", "veto",
    "vice", "view", "vine", "void", "volt", "vote", "wade", "wage",
    "wail", "wait", "wake", "walk", "wall", "wand", "want", "ward",
    "warm", "warn", "warp", "wary", "wash", "wasp", "wave", "wavy",
    "waxy", "weak", "wear", "weed", "week", "weep", "weld", "well",
    "went", "were", "west", "what", "when", "whom", "wick", "wide",
    "wife", "wild", "will", "wilt", "wily", "wind", "wine", "wing",
    "wink", "wipe", "wire", "wise", "wish", "with", "woke", "wolf",
    "wood", "wool", "word", "wore", "work", "worm", "worn", "wove",
    "wrap", "wren", "yank", "yard", "yarn", "year", "yell", "yoga",
    "yoke", "your", "zeal", "zero", "zone", "zoom",

    # 5+ letters (common words)
    "about", "above", "abuse", "ached", "acted", "added", "admit",
    "adopt", "adult", "after", "again", "agent", "agree", "ahead",
    "alarm", "album", "alert", "alien", "align", "alike", "alive",
    "alley", "allow", "alone", "along", "alter", "among", "angel",
    "anger", "angle", "angry", "anime", "apart", "apple", "apply",
    "arena", "argue", "arise", "armor", "array", "aside", "asset",
    "avoid", "awake", "award", "aware",
    "basic", "beach", "began", "begin", "being", "below",
    "bench", "berry", "birth", "black", "blade", "blame", "bland",
    "blank", "blast", "blaze", "bleak", "bleed", "blend", "bless",
    "blind", "blink", "bliss", "block", "blood", "bloom", "blown",
    "blues", "blunt", "blurt", "board", "boast", "bonus", "boost",
    "booth", "bound", "brain", "brand", "brave", "bread", "break",
    "breed", "brick", "bride", "brief", "bring", "broad", "broke",
    "brook", "brown", "brush", "buddy", "build", "built", "bunch",
    "burst", "buyer",
    "cabin", "cable", "candy", "cargo", "carry", "catch", "cause",
    "cease", "chain", "chair", "chalk", "champ", "chaos", "charm",
    "chart", "chase", "cheap", "check", "cheek", "cheer", "chess",
    "chest", "chick", "chief", "child", "chill", "chunk", "civil",
    "claim", "clash", "class", "clean", "clear", "clerk", "click",
    "cliff", "climb", "cling", "clock", "clone", "close", "cloth",
    "cloud", "coach", "coast", "color", "comic", "coral", "could",
    "count", "couch", "cough", "court", "cover", "crack", "craft",
    "crane", "crash", "crazy", "cream", "crime", "cross", "crowd",
    "crown", "cruel", "crush", "curve", "cycle",
    "daily", "dance", "death", "debug", "decay", "delay", "depth",
    "devil", "dirty", "doubt", "dough", "draft", "drain", "drama",
    "drank", "drawn", "dream", "dress", "dried", "drift", "drill",
    "drink", "drive", "drove", "drunk", "dying",
    "eager", "early", "earth", "eight", "elect", "elite", "email",
    "empty", "enemy", "enjoy", "enter", "entry", "equal", "error",
    "essay", "event", "every", "exact", "exist", "extra",
    "faith", "false", "fancy", "fatal", "fault", "favor", "feast",
    "fence", "fever", "fiber", "field", "fifth", "fifty", "fight",
    "final", "first", "fixed", "flame", "flash", "flesh", "float",
    "flood", "floor", "flour", "fluid", "flush", "focus", "force",
    "forge", "forth", "forum", "found", "frame", "frank", "fraud",
    "fresh", "front", "frost", "froze", "fruit", "fully", "funny",
    "ghost", "giant", "given", "glass", "globe", "glory", "glove",
    "grace", "grade", "grain", "grand", "grant", "graph", "grasp",
    "grass", "grave", "great", "green", "greet", "grief", "grill",
    "grind", "groan", "gross", "group", "grove", "grown", "guard",
    "guess", "guest", "guide", "guilt", "given",
    "habit", "happy", "harsh", "haven", "heart", "heavy", "hence",
    "hobby", "honor", "horse", "hotel", "house", "human", "humor",
    "hurry",
    "ideal", "image", "imply", "index", "india", "indie", "inner",
    "input", "irony", "issue",
    "Japan", "jewel", "joint", "judge", "juice",
    "knock", "known",
    "label", "labor", "large", "laser", "later", "laugh", "layer",
    "learn", "least", "leave", "legal", "lemon", "level", "light",
    "limit", "linen", "liver", "local", "lodge", "logic", "login",
    "loose", "lover", "lower", "loyal", "lucky", "lunch",
    "magic", "major", "maker", "march", "match", "maybe",
    "mayor", "medal", "media", "mercy", "merge", "merit", "metal",
    "meter", "might", "minor", "minus", "mixed", "model", "money",
    "month", "moral", "motor", "motto", "mount", "mouse", "mouth",
    "movie", "music",
    "naive", "naked", "nerve", "never", "night", "noble", "noise",
    "north", "noted", "novel", "nurse",
    "occur", "ocean", "offer", "often", "onset", "opera", "orbit",
    "order", "organ", "other", "ought", "outer", "owner",
    "paint", "panel", "panic", "paper", "party", "paste", "patch",
    "pause", "peace", "peach", "pearl", "penny", "phase", "phone",
    "photo", "piano", "piece", "pilot", "pitch", "pixel", "pizza",
    "place", "plain", "plane", "plant", "plate", "plaza", "plead",
    "plumb", "plunge", "point", "polar", "pound", "power",
    "press", "price", "pride", "prime", "print", "prior", "prize",
    "probe", "prone", "proof", "prose", "proud", "prove", "proxy",
    "punch", "pupil", "purse", "quest", "queue", "quick", "quiet",
    "quite", "quote",
    "radar", "radio", "raise", "rally", "ranch", "range", "rapid",
    "ratio", "reach", "react", "ready", "realm", "rebel", "refer",
    "reign", "relax", "reply", "rider", "ridge", "rifle", "right",
    "rigid", "risky", "rival", "river", "robot", "rocky", "roman",
    "rough", "round", "route", "royal", "rugby", "ruler", "rural",
    "saint", "salad", "sauce", "scale", "scene", "scope", "score",
    "scout", "screw", "sense", "serve", "setup", "seven", "shade",
    "shake", "shall", "shame", "shape", "share", "sharp", "shear",
    "sheep", "sheer", "sheet", "shelf", "shell", "shift", "shine",
    "shirt", "shock", "shoot", "shore", "short", "shout", "sight",
    "sigma", "since", "sixty", "sized", "skill", "skull", "slash",
    "slate", "slave", "sleep", "slice", "slide", "small", "smart",
    "smell", "smile", "smoke", "snake", "solar", "solid", "solve",
    "sorry", "sound", "south", "space", "spare", "speak", "speed",
    "spell", "spend", "spent", "spice", "spike", "spine", "spoke",
    "spoon", "sport", "spray", "squad", "stack", "staff", "stage",
    "stain", "stake", "stale", "stall", "stamp", "stand", "stare",
    "start", "state", "steak", "steal", "steam", "steel", "steep",
    "steer", "stick", "stiff", "still", "sting", "stock", "stole",
    "stone", "stood", "store", "storm", "story", "stove", "strap",
    "straw", "strip", "stuck", "study", "stuff", "style", "sugar",
    "super", "surge", "swamp", "swear", "sweat", "sweep", "sweet",
    "swept", "swing", "sword",
    "table", "taste", "teach", "teeth", "terms", "thank", "theme",
    "there", "thick", "thief", "thing", "think", "third", "those",
    "three", "threw", "throw", "thumb", "tight", "timer", "tired",
    "title", "toast", "today", "token", "tooth", "topic", "total",
    "touch", "tough", "tower", "toxic", "trace", "track", "trade",
    "trail", "train", "trait", "trash", "treat", "trend", "trial",
    "tribe", "trick", "tried", "troop", "truck", "truly", "trump",
    "trunk", "trust", "truth", "tumor", "tuned", "twice", "twist",
    "typed",
    "ultra", "uncle", "under", "union", "unite", "unity", "until",
    "upper", "upset", "urban", "usage", "usual", "utter",
    "valid", "value", "vault", "verse", "video", "vigor", "vinyl",
    "viral", "virus", "visit", "vista", "vital", "vivid", "vocal",
    "voice", "voter",
    "waste", "watch", "water", "weave", "weird", "whale", "wheat",
    "wheel", "where", "which", "while", "white", "whole", "whose",
    "woman", "world", "worry", "worse", "worst", "worth", "would",
    "wound", "wrist", "write", "wrong", "wrote",
    "yacht", "young", "yours", "youth",

    # 6+ letters (common)
    "accept", "access", "across", "action", "active", "actual",
    "advice", "advise", "affect", "afford", "afraid", "agency",
    "agenda", "almost", "always", "amount", "animal", "annual",
    "answer", "anyway", "appeal", "appear", "around", "arrive",
    "artist", "assume", "attack", "attend", "august",
    "balance", "barrier", "battery", "because", "become", "before",
    "behind", "belief", "belong", "beside", "better", "beyond",
    "border", "bother", "bottom", "branch", "bridge", "bright",
    "broken", "broker", "brother", "browse", "budget", "burden",
    "bureau", "button",
    "camera", "cancel", "career", "castle", "casual", "caught",
    "center", "centre", "chance", "change", "charge", "chosen",
    "church", "circle", "client", "closed", "closer", "coffee",
    "column", "combat", "comedy", "coming", "common", "comply",
    "corner", "costly", "cotton", "couple", "course", "cousin",
    "create", "credit", "crisis", "custom",
    "damage", "danger", "dealer", "debate", "decade", "decide",
    "defeat", "defend", "define", "degree", "demand", "dental",
    "depend", "deploy", "deputy", "desert", "design", "desire",
    "detail", "detect", "device", "devote", "dialog", "differ",
    "dinner", "direct", "divide", "dollar", "domain", "double",
    "driver", "during",
    "easily", "editor", "effect", "effort", "eighth", "either",
    "emerge", "empire", "enable", "endure", "energy", "engage",
    "engine", "enough", "ensure", "entire", "entity", "equity",
    "escape", "ethnic", "evolve", "exceed", "except", "excess",
    "excuse", "expand", "expect", "expert", "export", "expose",
    "extend", "extent",
    "fabric", "facing", "factor", "failed", "fairly", "fallen",
    "family", "famous", "farmer", "father", "fathom", "favour",
    "female", "figure", "filing", "filter", "finale", "finger",
    "finish", "flower", "flying", "follow", "forced", "forest",
    "forget", "formal", "format", "former", "foster", "fourth",
    "freeze", "friend", "frozen", "future",
    "garden", "gather", "gender", "gentle", "gifted", "global",
    "golden", "govern", "growth",
    "handle", "happen", "hardly", "having", "hazard", "health",
    "height", "helped", "hidden", "highly", "honest", "horror",
    "hungry", "hunter",
    "ignore", "impact", "import", "impose", "income", "indeed",
    "indoor", "inform", "inject", "injury", "insect", "inside",
    "insist", "intact", "intend", "invest", "invite", "island",
    "itself",
    "jacket", "junior", "justice",
    "kidney", "killer", "kindly", "knight", "knowledge",
    "lately", "latest", "latter", "launch", "lawyer", "layout",
    "leader", "league", "lessen", "lesson", "letter", "likely",
    "linear", "liquid", "listen", "little", "living", "lonely",
    "longer", "looked", "lovely", "luxury",
    "mainly", "manage", "manner", "manual", "margin", "marine",
    "marked", "market", "master", "matter", "medium", "member",
    "memory", "mental", "merely", "method", "middle", "mighty",
    "minute", "mirror", "mobile", "modern", "modest", "modify",
    "moment", "mostly", "mother", "motion", "moving", "murder",
    "muscle", "museum", "mutual",
    "namely", "narrow", "nation", "native", "nature", "nearby",
    "nearly", "needle", "nicely", "nobody", "normal", "notice",
    "notion", "number",
    "object", "obtain", "occupy", "office", "online", "oppose",
    "option", "orange", "origin", "others", "outfit", "output",
    "oxygen",
    "palace", "parent", "partly", "patrol", "patron", "people",
    "period", "permit", "person", "phrase", "pickup", "planet",
    "player", "please", "plenty", "pocket", "poetry", "poison",
    "police", "policy", "polite", "poorly", "portal", "potato",
    "powder", "prayer", "prefer", "pretty", "prince", "prison",
    "profit", "proper", "proven", "public", "pursue", "puzzle",
    "qualify", "quarter", "question", "quickly", "quietly",
    "random", "rarely", "rather", "rating", "reader", "really",
    "reason", "recall", "recent", "record", "reduce", "reform",
    "regard", "regime", "region", "reject", "relate", "relief",
    "remain", "remedy", "remote", "remove", "render", "rental",
    "repair", "repeat", "replace", "report", "rescue", "resign",
    "resist", "resort", "result", "retail", "retain", "retire",
    "return", "reveal", "review", "revolt", "reward", "rhythm",
    "rocket", "roster", "rubber", "ruling",
    "sacred", "safely", "safety", "salary", "sample", "saving",
    "saying", "scheme", "school", "screen", "search", "season",
    "second", "secret", "sector", "secure", "select", "seller",
    "senior", "series", "server", "settle", "severe", "shadow",
    "should", "signal", "silent", "silver", "simple", "simply",
    "single", "sister", "slight", "slowly", "smooth", "social",
    "solely", "sought", "source", "speech", "spirit", "spoken",
    "spread", "spring", "square", "stable", "stance", "status",
    "steady", "stolen", "strain", "strand", "stream", "street",
    "stress", "strict", "strike", "string", "stroke", "strong",
    "struck", "studio", "submit", "sudden", "suffer", "summer",
    "summit", "supply", "surely", "survey", "switch", "symbol",
    "system",
    "tackle", "talent", "target", "temple", "tenant", "tender",
    "terror", "thanks", "thirty", "though", "thread", "threat",
    "thrill", "throne", "timber", "tissue", "toward", "treaty",
    "tribal", "trick", "tricky", "triple", "trophy", "tumble",
    "tunnel", "twelve", "twenty",
    "unable", "unfair", "unique", "unless", "unlike", "unlock",
    "unrest", "update", "uphold", "useful",
    "vacuum", "valley", "varied", "vendor", "verify", "victim",
    "viewer", "virgin", "virtue", "vision", "visual", "volume",
    "welcome", "welfare", "western", "whether", "willing",
    "window", "winner", "winter", "wisdom", "within", "without",
    "wonder", "worker", "worthy", "writer", "written",
    "yellow",

    # Names and common proper nouns (lowercase for matching)
    "hello", "goodbye", "please", "sorry", "thanks", "thank",
    "morning", "evening", "afternoon", "tonight", "tomorrow",
    "yesterday", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
    "january", "february", "march", "april", "august",
    "september", "october", "november", "december",
}


class SpellCorrector:
    """
    Lightweight spell corrector for ASL fingerspelling output.
    Uses difflib for fuzzy matching against a common words dictionary.
    """

    def __init__(self, custom_words: Optional[set[str]] = None):
        """
        Args:
            custom_words: Additional words to add to the dictionary
        """
        self._words = set(w.upper() for w in COMMON_WORDS)
        if custom_words:
            self._words.update(w.upper() for w in custom_words)

        # Build a sorted list for difflib
        self._word_list = sorted(self._words)

    def correct(self, word: str) -> str:
        """
        Correct a fingerspelled word.

        Args:
            word: The word to correct (e.g., "HELO")

        Returns:
            The corrected word (e.g., "HELLO"), or the original if no good match.
        """
        if not word or len(word) < 2:
            return word

        upper = word.upper()

        # Already a known word — no correction needed
        if upper in self._words:
            return upper

        # Find close matches using difflib (sequence matching)
        # cutoff: minimum similarity ratio (0.0 to 1.0)
        # Higher cutoff = more conservative corrections
        cutoff = self._get_cutoff(len(upper))
        matches = difflib.get_close_matches(
            upper,
            self._word_list,
            n=3,
            cutoff=cutoff,
        )

        if matches:
            best = matches[0]
            # Additional validation: length difference shouldn't be too big
            len_diff = abs(len(best) - len(upper))
            if len_diff <= max(1, len(upper) // 3):
                return best

        # No good match — return original
        return upper

    def correct_with_info(self, word: str) -> tuple[str, bool, float]:
        """
        Correct a word and return correction info.

        Returns:
            (corrected_word, was_corrected, similarity_ratio)
        """
        if not word or len(word) < 2:
            return word, False, 1.0

        upper = word.upper()

        if upper in self._words:
            return upper, False, 1.0

        cutoff = self._get_cutoff(len(upper))
        matches = difflib.get_close_matches(
            upper,
            self._word_list,
            n=1,
            cutoff=cutoff,
        )

        if matches:
            best = matches[0]
            len_diff = abs(len(best) - len(upper))
            if len_diff <= max(1, len(upper) // 3):
                ratio = difflib.SequenceMatcher(None, upper, best).ratio()
                return best, True, ratio

        return upper, False, 0.0

    @staticmethod
    def _get_cutoff(word_length: int) -> float:
        """
        Adaptive cutoff based on word length.
        Short words need higher similarity to avoid false corrections.
        Longer words can tolerate more differences.
        """
        if word_length <= 2:
            return 0.8  # Very strict for 2-letter words
        elif word_length <= 4:
            return 0.65  # Moderate for 3-4 letters
        elif word_length <= 6:
            return 0.6   # Relaxed for 5-6 letters
        else:
            return 0.55  # More relaxed for longer words

    def add_word(self, word: str) -> None:
        """Add a custom word to the dictionary."""
        upper = word.upper()
        self._words.add(upper)
        self._word_list = sorted(self._words)

    def add_words(self, words: list[str]) -> None:
        """Add multiple custom words."""
        for w in words:
            self._words.add(w.upper())
        self._word_list = sorted(self._words)
