"""Phase-2 QAT corpus prompts (QUANT_LOG §9.4): ~640 prompts, mix 30% code / 20% agentic tool calls / 20% multilingual /
15% math+reasoning / 15% rare-token text. Deterministic (seed 7). Checked afterwards against all eval prompt sets."""
import json, random
R = random.Random(7)
out = []
def add(p, tools=False, cat=""):
    out.append({"p": p, "tools": tools, "cat": cat})

# ---- code (~190) -----------------------------------------------------------------------------------------------
langs = ["Python", "Rust", "Go", "C", "C++20", "Java", "Kotlin", "Swift", "TypeScript", "Haskell", "Elixir", "Zig",
         "Julia", "Lua", "Scala", "OCaml", "C#", "Ruby", "PHP", "Dart", "Fortran", "R", "Bash", "PowerShell", "Nim",
         "Clojure", "F#", "Perl", "MATLAB", "WGSL"]
tasks = ["a trie with insert, delete and prefix search", "a ring buffer with overwrite-on-full semantics",
         "a token-bucket rate limiter", "Dijkstra's algorithm on an adjacency list", "a union-find with path compression",
         "a tiny JSON tokenizer", "a Bloom filter with configurable false-positive rate", "a skip list",
         "run-length encoding and decoding", "a matrix multiplication tiled for cache", "a CSV parser that handles quoted fields",
         "an interval tree for overlap queries", "a topological sort that reports cycles", "a reservoir sampler",
         "a priority queue keyed by deadline", "a simple regex matcher supporting . and *", "Huffman coding",
         "a sparse vector dot product", "a k-means clustering step", "a URL router with path parameters",
         "an event emitter with once() and off()", "a retry helper with exponential backoff and jitter",
         "a consistent-hashing ring", "a Levenshtein distance with O(n) memory", "a simple arena allocator",
         "a markdown-to-HTML converter for headings and lists", "a fixed-point Q16.16 arithmetic type",
         "a quadtree for 2D point queries", "a minimal Forth interpreter", "a red-black tree insert"]
review = ["Explain what can go wrong with this lock ordering and fix it: thread A locks m1 then m2, thread B locks m2 then m1.",
          "Why does this leak memory in C? `char *s = malloc(64); s = realloc(s, 128); if (!s) return;` Fix it.",
          "This Rust code fails the borrow checker: `let first = &v[0]; v.push(4); println!(\"{}\", first);` Explain and fix.",
          "Why is `if (x = 5)` in C a bug that compiles, and how do compilers warn about it?",
          "Explain why this JavaScript prints 3 three times: `for (var i=0;i<3;i++) setTimeout(()=>console.log(i))`.",
          "What is wrong with comparing floats with == in a physics loop? Show a robust alternative.",
          "A Go service leaks goroutines when clients disconnect. Show the typical cause and the context-based fix.",
          "This Python class uses a mutable default argument `def __init__(self, tags=[])`. Explain the bug and fix it.",
          "Explain integer overflow in `int mid = (lo + hi) / 2;` and two correct alternatives.",
          "Why does this SQL return no rows: `SELECT * FROM t WHERE col = NULL`? Fix it and explain three-valued logic."]
concepts = ["copy-on-write", "tail-call optimization", "memory barriers on ARM", "the Rust drop order", "Python descriptors",
            "C++ move semantics", "Java's happens-before relation", "Go escape analysis", "JIT inline caches",
            "WebAssembly linear memory", "the ELF dynamic loader", "Linux io_uring", "epoll vs kqueue",
            "B+ tree page splits", "MVCC in PostgreSQL", "LSM-tree compaction", "CRDT counters", "Raft leader election",
            "TCP congestion control (CUBIC vs BBR)", "QUIC connection migration", "Unicode normalization forms NFC/NFD",
            "SIMD lane masking", "GPU warp divergence", "Metal threadgroup memory", "Apple Neural Engine weight palettization",
            "int4 weight-only quantization", "KV-cache paging", "rotary position embeddings", "flash attention tiling",
            "the Bellman equation", "Kalman filter predict/update", "bloom filters vs cuckoo filters"]
for t in R.sample(tasks, 30):
    for l in R.sample(langs, 3):
        add(f"Implement {t} in {l}, with a short usage example and its time complexity.", cat="code")
for r in review:
    add(r, cat="code")
for c in concepts:
    add(f"Explain {c} to an experienced engineer, with a small code or numeric example.", cat="code")
for l in R.sample(langs, 12):
    add(f"Write idiomatic {l} unit tests for a function `slugify(title)` that lowercases, strips accents and joins words with hyphens.", cat="code")

# ---- agentic tool calls (~130) ----------------------------------------------------------------------------------
files = ["src/server/routes.py", "lib/cache.ts", "cmd/worker/main.go", "Sources/App/Model.swift", "crates/core/src/lib.rs",
         "app/models/user.rb", "pkg/db/migrate.go", "frontend/src/hooks/useAuth.tsx", "scripts/release.sh",
         "include/ringbuf.h", "tests/test_parser.py", "docs/CONTRIBUTING.md", "Makefile", "pyproject.toml",
         "config/settings.yaml", "android/app/build.gradle", "infra/terraform/main.tf", ".github/workflows/ci.yml"]
acts = ["The CI job fails on {f}; find the cause and fix it.", "Add structured logging to {f} without changing behaviour.",
        "Refactor {f} to remove the duplicated error handling.", "Add a unit test that covers the empty-input path of {f}.",
        "Find all callers of the function defined at the top of {f} and update them to the new signature.",
        "Upgrade the dependency pinned in {f} and fix whatever breaks.", "Explain what {f} does, then add docstrings.",
        "There's a race condition reported in {f}. Investigate and propose a patch.",
        "Make {f} configurable through an environment variable with a sane default.",
        "Profile the slow path in {f} and optimize it."]
for a in acts:
    for f in R.sample(files, 11):
        add(a.format(f=f), tools=True, cat="tool")
for p in ["List the TODO comments in the repository and group them by file.", "Set up pre-commit hooks for formatting and linting.",
          "Bump the version to 3.4.0 everywhere it appears and update the changelog.", "Find dead code that is never imported.",
          "Write a migration that adds a non-null column with a default to the users table.",
          "The README's install steps are outdated; verify them against the code and fix them.",
          "Add a /metrics endpoint exposing request counts.", "Reproduce the bug from issue #412: dates shift by one day in UTC-8.",
          "Split the 2,000-line utils module into cohesive files.", "Replace print debugging with the logging module across the project.",
          "Add type hints to the public API and make mypy pass.", "Check the Dockerfile for size and security improvements.",
          "Add retries to the HTTP client wrapper and make them testable.", "Find where secrets might be logged and redact them.",
          "Port the Python CLI's argument parsing from argparse to click.", "Make the test suite run in parallel safely.",
          "Investigate why memory grows on every request in the worker.", "Add pagination to the /items listing endpoint.",
          "Generate a minimal reproducible example for the flaky test test_sync_timeout.", "Rename the package from `oldname` to `newname`."]:
    add(p, tools=True, cat="tool")

# ---- multilingual (~130) -----------------------------------------------------------------------------------------
ml = {
 "de": ["Erkläre, wie ein Wärmepumpe im Winter funktioniert und wann sie ineffizient wird.", "Schreibe eine höfliche E-Mail an den Vermieter wegen einer defekten Heizung.",
        "Was ist der Unterschied zwischen Föderalismus in Deutschland und in der Schweiz?", "Schreibe eine Python-Funktion, die deutsche Umlaute in ASCII umschreibt, und erkläre sie.",
        "Fasse die Ursachen der Inflation 2022 in Europa in fünf Punkten zusammen.", "Wie plane ich eine dreitägige Wanderung im Allgäu? Mit Packliste.",
        "Erkläre die Relativitätstheorie einem Zwölfjährigen.", "Löse: Ein Zug fährt 240 km in 2,5 Stunden. Wie schnell ist er im Durchschnitt, und wie lange braucht er für 400 km?"],
 "es": ["Explica la diferencia entre 'ser' y 'estar' con diez ejemplos.", "Escribe un cuento corto sobre un faro abandonado en Galicia.",
        "¿Cómo funciona la fotosíntesis a nivel molecular? Explícalo paso a paso.", "Escribe una función en JavaScript que valide un DNI español y explica el algoritmo de la letra.",
        "Resume las causas de la independencia de México.", "Dame un plan de entrenamiento de 8 semanas para correr 10 km.",
        "¿Qué es una hipoteca de tipo variable y qué riesgos tiene?", "Traduce al español y explica el sentido: 'The early bird catches the worm'."],
 "fr": ["Explique le fonctionnement du système électoral français à deux tours.", "Rédige une lettre de motivation pour un poste de développeur backend à Lyon.",
        "Quelle est la différence entre le passé composé et l'imparfait ? Donne des exemples.", "Écris une fonction Python qui convertit des nombres en toutes lettres en français (jusqu'à 999 999).",
        "Explique la théorie de l'évolution de Darwin et ses critiques historiques.", "Propose une recette de ratatouille et explique chaque étape.",
        "Comment fonctionne une centrale nucléaire à eau pressurisée ?", "Résous : la somme de trois entiers consécutifs vaut 162. Quels sont-ils ?"],
 "ru": ["Объясни, как работает блокчейн, простыми словами.", "Напиши короткое стихотворение о зимнем Петербурге.",
        "В чём разница между совершенным и несовершенным видом глагола? Приведи примеры.", "Напиши функцию на Python, которая транслитерирует русский текст латиницей по ГОСТ.",
        "Кратко перескажи сюжет «Преступления и наказания».", "Как подготовиться к собеседованию на позицию data scientist?",
        "Объясни принцип работы двигателя внутреннего сгорания.", "Реши задачу: в классе 30 учеников, 40% из них девочки. Сколько мальчиков?"],
 "ja": ["量子コンピュータの仕組みを高校生向けに説明してください。", "東京で三日間観光する旅程を提案してください。",
        "敬語の「尊敬語」と「謙譲語」の違いを例文付きで説明してください。", "Pythonで和暦を西暦に変換する関数を書いて、説明してください。",
        "俳句を五つ作って、それぞれの季語を説明してください。", "日本の少子高齢化の原因と対策をまとめてください。",
        "寿司の歴史を簡単に説明してください。", "次の問題を解いてください：ある数の3倍に7を足すと40になります。その数は？"],
 "zh": ["请解释Transformer模型中的注意力机制。", "写一篇关于春节习俗的短文。",
        "用Go语言实现一个并发安全的计数器，并解释为什么需要锁。", "比较唐朝和宋朝的经济发展特点。",
        "如何在家里种植番茄？请给出详细步骤。", "解释一下什么是边际效用递减，并举例。",
        "请帮我写一封辞职信，语气要礼貌。", "解题：一个长方形的周长是36厘米，长是宽的2倍，求面积。"],
 "ko": ["인공지능이 의료 분야에 미치는 영향을 설명해 주세요.", "서울에서 하루 여행 코스를 추천해 주세요.",
        "한국어의 존댓말과 반말의 차이를 예시와 함께 설명해 주세요.", "파이썬으로 한글 초성만 추출하는 함수를 작성하고 설명해 주세요.",
        "김치 담그는 방법을 단계별로 알려 주세요.", "다음 문제를 풀어 주세요: 사과 12개를 3명이 똑같이 나누면 한 명당 몇 개인가요?"],
 "pt": ["Explique como funciona o sistema de saúde SUS no Brasil.", "Escreva uma crônica curta sobre um dia de chuva em Lisboa.",
        "Qual a diferença entre 'por que', 'porque', 'por quê' e 'porquê'?", "Escreva uma função em Python que valide um CPF e explique o algoritmo.",
        "Resuma a história da independência do Brasil.", "Como montar um orçamento doméstico mensal?"],
 "it": ["Spiega la differenza tra il congiuntivo e l'indicativo con esempi.", "Scrivi una breve storia ambientata a Venezia durante il carnevale.",
        "Come funziona il sistema bicamerale italiano?", "Scrivi una funzione Rust che conti le parole in un testo italiano, gestendo gli apostrofi.",
        "Dammi la ricetta tradizionale della carbonara e spiega gli errori comuni.", "Risolvi: un negozio sconta del 20% un articolo da 85 euro. Quanto costa?"],
 "ar": ["اشرح كيف تعمل الطاقة الشمسية بطريقة مبسطة.", "اكتب قصة قصيرة عن صياد في الإسكندرية.",
        "ما الفرق بين الفعل اللازم والفعل المتعدي؟ أعط أمثلة.", "اكتب دالة بلغة بايثون تعكس نصاً عربياً مع الحفاظ على التشكيل، واشرحها.",
        "لخص أسباب ازدهار الحضارة الإسلامية في العصر العباسي.", "حل المسألة: إذا كان ثمن ٣ كتب ٤٥ ريالاً، فكم ثمن ٧ كتب؟"],
 "hi": ["मशीन लर्निंग और डीप लर्निंग में क्या अंतर है? उदाहरण सहित समझाइए।", "दिवाली के त्योहार पर एक छोटा निबंध लिखिए।",
        "पायथन में एक फ़ंक्शन लिखिए जो किसी संख्या को हिंदी शब्दों में बदले।", "इस प्रश्न को हल कीजिए: एक कार 60 किमी/घंटा की गति से 3.5 घंटे चलती है। कुल दूरी कितनी है?"],
 "tr": ["Osmanlı İmparatorluğu'nun yükselme dönemini özetle.", "Python'da Türkçe karakterleri doğru sıralayan bir fonksiyon yaz ve açıkla.",
        "İstanbul'da üç günlük bir gezi planı hazırla.", "Şu problemi çöz: Bir işi Ali 6 günde, Veli 12 günde bitiriyor. Birlikte kaç günde biterler?"],
 "pl": ["Wyjaśnij, jak działa szczepionka mRNA.", "Napisz funkcję w Pythonie, która odmienia polskie liczebniki przez przypadki (dla 1-10), i wyjaśnij ją.",
        "Opisz historię Krakowa w pięciu akapitach.", "Rozwiąż: pociąg jedzie 90 km/h przez 2 godziny i 20 minut. Jaką drogę pokona?"],
 "vi": ["Giải thích cách hoạt động của mạng 5G.", "Viết một đoạn văn ngắn về Tết Nguyên Đán ở Việt Nam.",
        "Viết hàm Python bỏ dấu tiếng Việt và giải thích.", "Giải bài toán: Một hình tròn có bán kính 7 cm. Tính diện tích."],
 "uk": ["Поясни, як працює енергосистема під час пікових навантажень.", "Напиши функцію на Python для перевірки українського IBAN і поясни її."],
 "nl": ["Leg uit hoe de Deltawerken Nederland beschermen tegen overstromingen.", "Schrijf een Python-functie die Nederlandse postcodes valideert en leg haar uit."],
 "sv": ["Förklara hur allemansrätten fungerar i Sverige.", "Skriv en Python-funktion som validerar ett svenskt personnummer och förklara Luhn-algoritmen."],
 "id": ["Jelaskan cara kerja sistem pemilu di Indonesia.", "Tulis fungsi Python untuk memformat angka ke format Rupiah dan jelaskan."],
 "he": ["הסבר כיצד פועלת מערכת כיפת ברזל באופן כללי.", "כתוב פונקציה בפייתון שהופכת מספר לגימטריה ותסביר אותה."],
 "th": ["อธิบายการทำงานของระบบรถไฟฟ้า BTS ในกรุงเทพฯ", "เขียนฟังก์ชัน Python ที่ตัดคำภาษาไทยอย่างง่ายและอธิบาย"],
 "el": ["Εξήγησε τη δημοκρατία της αρχαίας Αθήνας.", "Γράψε μια συνάρτηση Python που μετατρέπει ελληνικό κείμενο σε greeklish."],
}
for lang, ps in ml.items():
    for p in ps:
        add(p, cat=f"ml_{lang}")
en_to = ["German", "Spanish", "French", "Russian", "Japanese", "Chinese", "Korean", "Portuguese", "Italian", "Arabic", "Hindi", "Turkish"]
for c in R.sample(concepts, 12):
    for lang in R.sample(en_to, 2):
        add(f"Explain {c} in {lang}. Keep technical terms in English in parentheses.", cat="ml_mixed")

# ---- math / reasoning (~100) ----------------------------------------------------------------------------------
for _ in range(25):
    a, b, c = R.randint(3, 19), R.randint(20, 90), R.randint(2, 9)
    add(f"A tank holds {b*10} liters. Pipe A fills it in {a} hours, pipe B drains it in {a+c} hours. Starting empty with both open, when is it full? Show your reasoning.", cat="math")
for _ in range(20):
    n, k = R.randint(8, 30), R.randint(2, 6)
    add(f"How many subsets of {{1, ..., {n}}} of size {k} contain no two consecutive integers? Derive a formula and evaluate it.", cat="math")
for _ in range(15):
    p = R.choice([0.1, 0.15, 0.2, 0.3, 0.35]); m = R.randint(5, 20)
    add(f"Each of {m} independent servers fails with probability {p} per day. What is the probability that at least two fail on a given day? Also give the expected number of failures.", cat="math")
for e in ["Prove that there are infinitely many primes of the form 4k+3.", "Show that every tree with n vertices has n-1 edges.",
          "Prove by induction that 1^3 + 2^3 + ... + n^3 = (n(n+1)/2)^2.", "Compute the integral of x^2 e^{-x} from 0 to infinity and explain each step.",
          "Find all real solutions of x^4 - 5x^2 + 4 = 0.", "What is the determinant of an n×n matrix with 2 on the diagonal and 1 elsewhere?",
          "Explain why the harmonic series diverges, with two different proofs.", "A 4-digit PIN has no repeated digits. How many PINs are there and what fraction are even?",
          "Three people check hats; hats are returned randomly. What's the probability nobody gets their own hat? Generalize to n.",
          "Is 2^61 - 1 prime? Explain how one would check it efficiently.", "Find the eigenvalues of [[2,1],[1,2]] and interpret them geometrically.",
          "A ladder 10 m long slides down a wall at 1 m/s at the top. How fast is the bottom moving when the top is 6 m high?",
          "Solve the recurrence T(n) = 3T(n/2) + n and explain the master theorem case.", "Estimate how many piano tuners work in a city of 2 million people (Fermi estimate).",
          "Five pirates divide 100 coins with the classic seniority voting rule. What happens? Explain the backward induction.",
          "Knights always tell the truth and knaves always lie. A says 'B is a knave', B says 'A and I are the same type'. Who is what?",
          "You have 12 coins, one counterfeit (heavier or lighter). Find it in 3 weighings and explain the strategy.",
          "Compute 17^2025 mod 23 and explain the method.", "What is the probability that a random chord of a circle is longer than the side of the inscribed equilateral triangle? Discuss Bertrand's paradox.",
          "A drawer has 10 red and 10 blue socks. In the dark, how many must you take to guarantee a matching pair? A pair of each color?"]:
    add(e, cat="math")

# ---- rare-token / structured text (~100) ------------------------------------------------------------------------
names = ["Nguyễn Thị Minh Khai", "Søren Kierkegaard", "Bjørk Guðmundsdóttir", "Krzysztof Penderecki", "Oluwaseun Adébáyọ̀",
         "Siddhartha Mukherjee", "Ngũgĩ wa Thiong'o", "Žofie Dvořáková", "Mahmoud Darwish", "Iñaki Etxeberria", "Aleksandr Solzhenitsyn",
         "Thích Nhất Hạnh", "Chimamanda Ngozi Adichie", "Jürgen Habermas", "Vytautas Landsbergis"]
for n in names:
    add(f"Write a short, factual-style biography card for {n} as YAML with fields name, native_script, born, fields, notable_works (use 'unknown' where unsure).", cat="rare")
for _ in range(15):
    add(f"Generate a JSON array of {R.randint(4, 9)} fake but realistic API log records with UUIDv4 ids, ISO-8601 timestamps with offsets, IPv6 addresses, HTTP methods, status codes and latency in ms. Then compute the p95 latency.", cat="rare")
for p in ["Write the LaTeX for the Navier–Stokes equations in vector form and explain each term.",
          "Write LaTeX for a 3×3 block matrix with Greek-letter entries and an aligned derivation of its determinant.",
          "Decode this base64 and explain what it is: aGVsbG8gd29ybGQsIHF1YW50aXplZCBtb2RlbHMh",
          "Explain this hex dump of a PNG header: 89 50 4E 47 0D 0A 1A 0A 00 00 00 0D 49 48 44 52.",
          "Write a regex that matches RFC 5322-ish email addresses, then list 8 tricky test strings and whether each matches.",
          "Explain the SMILES string CC(=O)Oc1ccccc1C(=O)O and draw its structure in ASCII.",
          "Write a Dockerfile, a docker-compose.yml and a .env.example for a Postgres + Redis + FastAPI stack.",
          "Write a Kubernetes Deployment and Service YAML with resource limits, liveness probes and a ConfigMap mount.",
          "Write a GraphQL schema for a library system and three example queries with variables.",
          "Write a protobuf definition for a telemetry message with nested enums and oneof, then show its JSON mapping.",
          "Convert these coordinates to decimal degrees: 48°51′29.6″N 2°17′40.2″E and 33°51′35.9″S 151°12′40.0″E.",
          "List the first 20 elements with symbol, atomic number and standard atomic weight as a Markdown table.",
          "Write an nginx config with TLS, HTTP/2, gzip, and a reverse proxy to localhost:8080 with websockets.",
          "Write a crontab with 6 entries using unusual schedules and explain each field.",
          "Show the git commands and the resulting `git log --graph --oneline` for a feature branch rebased onto main with a fixup.",
          "Write a SQL schema with CHECK constraints, a partial index and a generated column for an invoices table (PostgreSQL).",
          "Write a Makefile-style table of 15 common Unicode characters: code point, UTF-8 bytes in hex, name, e.g. U+00E9, U+20AC, U+1F600.",
          "Explain these semver ranges and which versions they match: ^1.2.3, ~0.4.1, >=2.0.0-rc.1 <3, 1.x || >=4.5.",
          "Generate 10 realistic but fake IBANs from different countries, explain the checksum, and verify one by hand.",
          "Write an OpenAPI 3.1 YAML for a /v2/orders/{orderId}/refunds endpoint with auth, errors and examples.",
          "Translate this chess game into commentary: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 6.Be3 e5 7.Nb3 Be6 8.f3 h5.",
          "Explain the musical chord symbols Cmaj7♯11, F♯m7♭5, B♭7alt and G/B with their notes.",
          "Write a short README in Markdown with badges, a table, a mermaid diagram and collapsible details sections.",
          "Write a TOML config for a Rust project with workspace members, features, profiles and target-specific dependencies.",
          "Write an x86-64 assembly (AT&T syntax) function that sums an int32 array, and explain the calling convention.",
          "Write ARM64 assembly for strlen and explain each instruction.",
          "Write a Vim regex substitution that swaps 'Last, First' to 'First Last' across a file, and the sed equivalent.",
          "Explain the IPA transcription of 'thoroughly', 'colonel', 'Worcestershire' and 'Leicester' in British English.",
          "Create a table of 12 airport IATA codes, ICAO codes, cities and UTC offsets.",
          "Write a jq command pipeline that flattens nested JSON, filters by a date range and outputs CSV.",
          "Write an SVG of a simple bar chart by hand with axes, labels and 5 bars.",
          "Write a GLSL fragment shader for a signed-distance-field rounded rectangle with anti-aliasing.",
          "Write Emoji-rich release notes 🚀✨🐛 for a mobile app, with a short changelog table.",
          "Explain this cron-like systemd timer: OnCalendar=Mon..Fri *-*-* 09:30:00 Europe/Berlin and write the unit files.",
          "Explain the BibTeX entries for an article, inproceedings and a thesis with all common fields.",
          "Give the HTTP/1.1 raw request and response for a CORS preflight, with all relevant headers.",
          "Write a Terraform module for an S3 bucket with versioning, lifecycle rules and a bucket policy.",
          "Write Wolfram Language code to plot a parametric surface and compute its area.",
          "Write a Prolog program for family relations with ancestor/2 and explain resolution on a query.",
          "Show a sample `strace -f` output of a program opening a file and explain each syscall line.",
          "Explain these Linux permission strings and octal modes: drwxr-sr-x, -rwsr-xr-x, 1777, 0640.",
          "Write a tiny Brainfuck program that prints 'Hi' and explain the tape at each step.",
          "Write a Mermaid sequence diagram for an OAuth 2.0 authorization code flow with PKCE.",
          "Write an ASCII table comparing time zones: UTC, PST, EST, CET, IST, JST, AEST with offsets and DST rules.",
          "Explain Morse code for 'SOS QUANT 27B' and the timing rules.",
          "Write a Haskell type signature puzzle with Functor/Applicative/Monad instances for a parser type and solve it.",
          "Write a GitHub Actions workflow with a matrix over OS and Python versions, caching and artifact upload.",
          "Explain the fields of a JWT and decode this header and payload: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI0MiIsImV4cCI6MTk5OTk5OTk5OX0.",
          "Write a Lean 4 proof that addition of natural numbers is commutative, with comments.",
          "Convert 1,234,567.891 to scientific notation, IEEE-754 single-precision hex, and Roman numerals for the integer part."]:
    add(p, cat="rare")

R.shuffle(out)
# This builds the unfiltered candidate pool, not the checked release prompt list.
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description="Build the seed-7 QAT candidate prompts before overlap filtering.")
parser.add_argument("--output", type=Path, required=True, help="New candidate JSON path; keep separate from the release list")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as stream:
    json.dump(out, stream, ensure_ascii=False, indent=0)
from collections import Counter
print(len(out), Counter(o["cat"].split("_")[0] for o in out))
