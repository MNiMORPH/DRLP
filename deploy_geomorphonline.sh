#!/usr/bin/env bash
#
# Build drlp_panel into GeomorphOnline's shared app directory and verify it.
#
# Everything here is LOCAL. It never commits and never pushes: publishing is a
# push to a shared remote and is a separate, deliberate act.
#
# Every check below is one the apps/README.md in that directory asks for, each
# recording a failure that has actually happened there. Read it before changing
# this script.
#
#   usage:  ./deploy_geomorphonline.sh [--bump] [path-to-GeomorphOnline.github.io]
#
# NOTE for anyone writing commit messages around this script: pass them with
# `git commit -F <file>`, never `-m "..."`. A double-quoted -m runs backticks
# as command substitution, which silently ate the word it was quoting out of a
# commit message on 2026-10-28 ("takes a  span").
#
# --bump increments the dev version first, which is what makes iterating on a
# live demo work at all. A rebuilt wheel keeps its filename, GitHub Pages
# serves wheels with max-age=600, and micropip will not take a ?v= query on a
# wheel URL -- so without a new filename a reader who loaded the page in the
# last ten minutes reinstalls the OLD wheel and runs the old model, with
# nothing anywhere to say so. The version is the cache-buster, not a release
# ceremony: bump it freely.
#
set -euo pipefail

DRLP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUMP=0
if [ "${1:-}" = "--bump" ]; then BUMP=1; shift; fi
SITE="${1:-$HOME/models/GeomorphOnline.github.io}"
APPS="$SITE/exercises/apps"
APP="interactive_demo/drlp_panel.py"

[ -d "$APPS" ] || { echo "no such app directory: $APPS" >&2; exit 1; }

say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- provenance
if [ "$BUMP" -eq 1 ]; then
    say "bump"
    python3 - <<'PYBUMP'
import re, pathlib
p = pathlib.Path("src/drlp/_version.py")
s = p.read_text()
m = re.search(r'__version__ = "(.*?)"', s)
old = m.group(1)
dev = re.match(r"(.*\.dev)(\d+)$", old)
if dev:
    new = f"{dev.group(1)}{int(dev.group(2)) + 1}"
else:
    new = old + ".dev1"
p.write_text(s.replace(f'"{old}"', f'"{new}"'))
print(f"   {old} -> {new}")
PYBUMP
    git add src/drlp/_version.py
    git commit -q -m "Bump version for a GeomorphOnline redeploy

A rebuilt wheel keeps its filename and GitHub Pages serves wheels with
max-age=600, so without a new filename a reader who has loaded the page in
the last ten minutes reinstalls the old wheel and silently runs the old
model. The version is the cache-buster.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011qNGd13MDBX6c4DdmdPFcH"
    echo "   committed"
fi

say "provenance"
cd "$DRLP"
DIRTY="$(git status --porcelain | wc -l)"
COMMIT="$(git rev-parse --short HEAD)"
VERSION="$(python3 -c 'import drlp; print(drlp.__version__)')"
REMOTE="$(git remote -v | head -1 || true)"
echo "   DRLP commit     $COMMIT   modified files: $DIRTY"
echo "   drlp version    $VERSION"
echo "   remote          ${REMOTE:-NONE -- this commit is not fetchable by anyone else}"
if [ "$DIRTY" -ne 0 ]; then
    echo "   REFUSING: build from a clean tree, so the commit identifies the code." >&2
    echo "   (A wheel built from a modified tree carries the same version as the" >&2
    echo "    release it branched from and is indistinguishable by filename.)" >&2
    exit 1
fi

# A rebuilt wheel keeps its name and browsers keep the old one, so the version
# must move on every deploy. Record what is already there to compare against.
PREV_WHEEL="$(ls "$APPS"/drlp-*.whl 2>/dev/null || true)"
if [ -n "$PREV_WHEEL" ]; then
    echo "   already deployed: $(basename "$PREV_WHEEL")"
    if [ "$(basename "$PREV_WHEEL")" = "drlp-$VERSION-py3-none-any.whl" ]; then
        echo "   REFUSING: the deployed wheel has this exact filename already." >&2
        echo "   Bump src/drlp/_version.py first. GitHub Pages serves wheels with" >&2
        echo "   max-age=600, so a returning reader reinstalls the OLD wheel under" >&2
        echo "   the same name and silently runs the old model." >&2
        exit 1
    fi
fi

# --------------------------------------------------- shared files, before
say "shared files before the build"
declare -A BEFORE
for f in artesian-embed.js artesian-embed.css \
         panel-1.9.4-py3-none-any.whl bokeh-3.9.2-py3-none-any.whl; do
    BEFORE[$f]="$(sha1sum "$APPS/$f" | cut -c1-40)"
    printf '   %-40s %s\n' "$f" "${BEFORE[$f]:0:8}"
done

# ------------------------------------------------------------------- build
# ONE -p: drlp_panel hand-rolls its own play/pause and does not import
# artesian.live, so it is grlp_panel's case rather than hillcreep's. A second
# -p would ship the artesian wheel the page never installs, and would rewrite
# that shared wheel for nothing.
#
# BOTH strip flags, every time. artesian re-downloads panel and bokeh on each
# build; omitting them replaces 11.5 MB of shared wheels with 36.9 MB of full
# ones, with nothing to warn you and every reader's first visit three times
# larger.
say "build"
artesian build "$APP" -o "$APPS" -p . -r numpy -r scipy \
    --strip-wheels --strip-vendored

# ------------------------------------------------------------------ verify
say "verify: the page's own install list"
INSTALL="$(grep -o 'micropip\.install(\[[^]]*\])' "$APPS/drlp_panel.js")"
echo "   $INSTALL"
for need in drlp- numpy scipy; do
    case "$INSTALL" in
        *"$need"*) echo "   ok    $need present" ;;
        *) echo "   FAIL  $need missing from the install list" >&2; exit 1 ;;
    esac
done

say "verify: the strip flags took"
ls -l "$APPS"/*.whl | awk '{printf "   %9.2f MB  %s\n", $5/1048576, $9}'
PANEL_MB="$(stat -c%s "$APPS/panel-1.9.4-py3-none-any.whl" | awk '{print int($1/1048576)}')"
if [ "$PANEL_MB" -gt 12 ]; then
    echo "   FAIL  panel wheel is ${PANEL_MB} MB: the strip flags did not apply." >&2
    exit 1
fi
echo "   ok    panel wheel ${PANEL_MB} MB (stripped; unstripped is 30)"

say "verify: did the build change the shared embed files?"
BUMP=0
for f in artesian-embed.js artesian-embed.css; do
    after="$(sha1sum "$APPS/$f" | cut -c1-40)"
    if [ "$after" != "${BEFORE[$f]}" ]; then
        echo "   CHANGED  $f  ${BEFORE[$f]:0:8} -> ${after:0:8}"
        echo "            bump ?v= in EVERY page under _pages/exercises/"
        BUMP=1
    else
        echo "   same     $f  ${after:0:8}"
    fi
done
# artesian re-downloads and re-hosts panel and bokeh on every build, which
# rewrites them with identical contents and different zip timestamps. Verify
# that is all it is, then put the committed bytes back, so a deploy of one app
# does not churn 11 MB of shared wheels through git history.
for f in panel-1.9.4-py3-none-any.whl bokeh-3.9.2-py3-none-any.whl; do
    after="$(sha1sum "$APPS/$f" | cut -c1-40)"
    if [ "$after" = "${BEFORE[$f]}" ]; then
        echo "   same     $f  (shared wheel untouched)"
        continue
    fi
    tmp="$(mktemp -d)"
    mkdir "$tmp/old" "$tmp/new"
    ( cd "$SITE" && git show "HEAD:exercises/apps/$f" ) > "$tmp/old.whl"
    ( cd "$tmp/old" && unzip -qq "$tmp/old.whl" )
    ( cd "$tmp/new" && unzip -qq "$APPS/$f" )
    if diff -r "$tmp/old" "$tmp/new" >/dev/null 2>&1; then
        cp "$APPS/$f" "$tmp/rebuilt-$f"          # recoverable copy first
        ( cd "$SITE" && git show "HEAD:exercises/apps/$f" ) > "$APPS/$f"
        echo "   restored $f  (contents identical, zip metadata only;"
        echo "            rebuilt copy kept at $tmp/rebuilt-$f)"
    else
        echo "   CHANGED  $f  -- CONTENTS differ, not just metadata. Stop and look." >&2
        diff -rq "$tmp/old" "$tmp/new" | head -5 >&2
        exit 1
    fi
done

say "what a cold reader downloads"
du -cb "$APPS"/*.whl "$APPS"/drlp_panel.* "$APPS"/artesian-embed.* 2>/dev/null \
    | tail -1 | awk '{printf "   %.2f MB self-hosted, plus the Pyodide runtime from its CDN\n", $1/1048576}'

say "left to do by hand"
echo "   1. open the page in a BROWSER and drive it. Nothing above does that,"
echo "      and it is the only check that catches a demo that compiles and"
echo "      then dies in the reader's console."
echo "   2. measure the demo's height at 900 px and set it on the iframe:"
echo "      it is what a reader looks at while ~60 MB of Pyodide arrives."
[ "$BUMP" -eq 1 ] && echo "   3. bump the ?v= hashes above in every exercise page."
echo "   $([ "$BUMP" -eq 1 ] && echo 4 || echo 3). git add / commit in the site repo, then push."
echo "      Publishing is yours: this script never commits and never pushes."
