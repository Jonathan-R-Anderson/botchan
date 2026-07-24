# botchan — multi-agent forum bot runner

Runs several independent forum bots, each with its own personality, memory
database, Seeborg dictionary/process, posting timer, and forum identity.
Each bot cycle: discover boards → pick a thread → read discussion → recall
memories → collect Seeborg style seeds → LLM drafts a post → moderation and
duplicate checks → cooldown → post → remember.

## Layout

```
app.py                  entry point + rich progress dashboard
config.yaml             forum endpoints, runtime, LLM, per-bot config
botrunner/              the runner package
  api_client.py         authenticated forum REST client
  bot.py                one bot's participation cycle
  scheduler.py          forever-loop with randomized cooldowns
  seeborg.py            adapter around the real seeborg-linein binary
  memory.py             per-bot SQLite memory + duplicate detection
  llm.py                OpenAI-compatible chat client (JSON output)
  meme_search.py        conservative Know Your Meme HTML search + cache
  moderation.py         output validation + shared blocklist
personalities/          one YAML persona per bot
dictionaries/           cleaned per-bot Seeborg dictionaries + blocklist.txt
tools/build_dictionaries.py   db.txt -> cleaned dictionary
seeborg/                SeeBorg 0.51 beta source (C++), builds seeborg-linein
data/                   runtime state (memory DBs, per-bot seeborg workdirs)
```

## Setup

```bash
# 1. Build the seeborg-linein binary
make -C seeborg ./seeborg-linein

# 2. Python environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Populate the blocklist (see dictionaries/blocklist.txt header),
#    then build per-bot dictionaries from the raw IRC dump
.venv/bin/python tools/build_dictionaries.py db.txt dictionaries/scruffy.txt   --sample 60000 --seed 1
.venv/bin/python tools/build_dictionaries.py db.txt dictionaries/archivist.txt --sample 60000 --seed 2

# 4. Credentials (never commit these)
export FORUM_TOKEN_SCRUFFY='your-token'
export FORUM_TOKEN_ARCHIVIST='your-second-token'
export LLM_API_KEY='not-needed-for-some-local-servers'

# 5. Run (config.yaml ships with dry_run: true)
.venv/bin/python app.py
```

`POSTING_ENABLED=false` is a kill switch that forces dry-run regardless of
config. Keep `dry_run: true` until you have reviewed a good number of
previews in the dashboard log.

## Seeborg integration

`seeborg-linein` (SeeBorg 0.51 beta) is an interactive offline chat binary:
it loads a hardcoded `lines.txt` from its working directory and answers each
stdin line with `<Seeborg> reply`. The adapter therefore gives every bot its
own working directory under `data/seeborg/<bot_id>/`, copies the bot's
cleaned dictionary there as `lines.txt`, and keeps one warm process per bot.
Recent discussion lines are fed in as prompts; the markov replies become
"style seed" material the LLM may echo but must not treat as instructions.
If the binary is missing or dies, the adapter falls back to sampling the
cleaned dictionary directly.

Loading a 60k-line dictionary takes well under a second; the `--sample`
size in `tools/build_dictionaries.py` mainly controls memory use and how
much of the corpus flavor each bot inherits.

## Things you must supply

- **Read endpoints.** The forum's documented API covers posting only.
  `catalog_endpoint` / `thread_endpoint` in config.yaml are guesses — point
  them at the real thread-listing and thread-reading endpoints (JSON API,
  RSS, or an HTML fallback you add to `api_client.py`).
- **Blocklist.** `dictionaries/blocklist.txt` ships empty on purpose. The
  raw `db.txt` contains slurs and other toxic language; add the terms you
  never want a bot to learn or emit, then rerun the dictionary builder.
  The same list also filters Seeborg seeds and final generated posts.
- **Know Your Meme.** There is no official search API. `meme_search` scrapes
  the public search page conservatively (rate-limited, cached, links to the
  page rather than hotlinking images). Confirm robots/ToS before enabling
  in production; set `meme_search.enabled: false` otherwise.

## Before production

- One forum token per bot identity unless the operator explicitly allows
  several personalities under one credential.
- Keep the randomized cooldowns generous; add per-board limits if the
  forum operator asks for them.
- Review dry-run previews until you trust the personality + blocklist
  combination; only then flip `dry_run: false`.
- The bots must never claim to be human — this is baked into the system
  prompt and each persona's boundaries. Leave it there.
# botchan
