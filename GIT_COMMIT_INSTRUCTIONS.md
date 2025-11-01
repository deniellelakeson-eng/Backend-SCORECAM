# Quick Git Commit Instructions

## Add Modified Files to `herbascan-backend` Repository

```bash
# Navigate to backend directory (or herbascan-backend repo root)
cd backend

# Check what files changed
git status

# Add all modified files
git add main.py Dockerfile requirements.txt

# Commit changes
git commit -m "Fix Railway deployment: improve logging and Railway Volume support"

# Push to repository
git push origin main
```

**Done!** Railway will automatically redeploy with your changes.

---

## Alternative: Add All Modified Files

```bash
cd backend
git add .
git commit -m "Fix Railway deployment: improve logging and Railway Volume support"
git push origin main
```


