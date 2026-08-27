#!/usr/bin/env bash
# One-time Fly.io provisioning for Lens OS API.
# Run ONCE on first deploy. Subsequent deploys: make fly-deploy
#
# Prerequisites:
#   1. flyctl installed — curl -L https://fly.io/install.sh | sh
#   2. fly auth login
#   3. A Neon Postgres (free tier, pgvector enabled out of the box):
#      https://neon.tech → New project → copy the connection string
#      Format: postgresql://user:pass@host.neon.tech/dbname?sslmode=require

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}==>${NC} $*"; }
ok()    { echo -e "${GREEN} ✓${NC}  $*"; }
die()   { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }

echo ""
echo "  Lens OS — Fly.io first-time setup"
echo "  ──────────────────────────────────"
echo ""

# ── 1. Preflight ─────────────────────────────────────────────────────────────
command -v fly &>/dev/null || die "flyctl not found. Install: curl -L https://fly.io/install.sh | sh"
fly auth whoami &>/dev/null || die "Not logged in. Run: fly auth login"
ok "flyctl authenticated"

# ── 2. Create app (idempotent) ────────────────────────────────────────────────
APP="lens-os-api"
info "Creating app '${APP}' (skipped if already exists)…"
fly apps create "$APP" 2>/dev/null && ok "App created" || ok "App already exists — continuing"

# ── 3. Collect secrets ────────────────────────────────────────────────────────
echo ""
info "Collecting secrets (input is hidden for sensitive values):"
echo ""

read -rp    "  GOOGLE_API_KEY  (required): " GOOGLE_API_KEY
[[ -z "$GOOGLE_API_KEY" ]] && die "GOOGLE_API_KEY is required"

read -rp    "  DATABASE_URL    (Neon connection string, required): " DATABASE_URL
[[ -z "$DATABASE_URL" ]]   && die "DATABASE_URL is required"

echo -n "  LENS_API_KEY    (leave blank to generate one): "
read -rs LENS_API_KEY; echo
if [[ -z "$LENS_API_KEY" ]]; then
    LENS_API_KEY="lens-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
    echo "  Generated: ${LENS_API_KEY}"
fi

read -rp    "  TAVILY_API_KEY  (optional, Enter to skip): " TAVILY_API_KEY || true

# ── 4. Set secrets via stdin (never touches shell history) ────────────────────
info "Setting secrets…"
{
    echo "GOOGLE_API_KEY=${GOOGLE_API_KEY}"
    echo "DATABASE_URL=${DATABASE_URL}"
    echo "LENS_API_KEY=${LENS_API_KEY}"
    [[ -n "${TAVILY_API_KEY:-}" ]] && echo "TAVILY_API_KEY=${TAVILY_API_KEY}"
} | fly secrets import --app "$APP"
ok "Secrets set"

# ── 5. Deploy (release_command runs init_db before traffic switches) ──────────
echo ""
info "Deploying (DB schema is applied automatically via release_command)…"
fly deploy --app "$APP"

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
ok "Deploy complete"
fly status --app "$APP"
echo ""
HOSTNAME=$(fly info --app "$APP" --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Hostname','<see above>'))" 2>/dev/null || echo "<see above>")
echo -e "  API endpoint:  ${GREEN}https://${HOSTNAME}${NC}"
echo "  Health check:  curl https://${HOSTNAME}/health"
echo ""
echo "  Next deploys:  make fly-deploy"
echo "  Logs:          fly logs --app ${APP}"
echo ""
