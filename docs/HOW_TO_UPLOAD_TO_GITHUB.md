# How to Upload This Project to GitHub

A step-by-step guide for uploading the AI Code Review project to GitHub
so it satisfies your German university MS application requirement.

---

## Step 1 — Create a GitHub Account (if you don't have one)

1. Go to https://github.com
2. Click **Sign up** → enter your email, create a password, choose a username
3. Verify your email address

---

## Step 2 — Create a New Repository on GitHub

1. Log in to GitHub
2. Click the **+** icon (top-right) → **New repository**
3. Fill in:
   - **Repository name:** `ai-code-review`
   - **Description:** `AI-Powered Code Review and Bug Prediction System — IMSciences BCS FYP 2021–2025`
   - **Visibility:** ✅ **Public** (required so the university can view it)
   - **DO NOT** tick "Add a README file" (we already have one)
4. Click **Create repository**

---

## Step 3 — Install Git on Your Computer

### Windows
Download from https://git-scm.com/download/windows and install with defaults.

### macOS
```bash
brew install git
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update && sudo apt install git -y
```

---

## Step 4 — Extract the Project Files

Unzip the downloaded `ai-code-review.zip` to a folder, e.g.:
- Windows: `C:\Projects\ai-code-review\`
- Mac/Linux: `~/Projects/ai-code-review/`

---

## Step 5 — Open a Terminal and Navigate to the Project

**Windows:** Right-click inside the `ai-code-review` folder → **Open in Terminal** (or Git Bash)

**Mac/Linux:**
```bash
cd ~/Projects/ai-code-review
```

---

## Step 6 — Configure Git (first time only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

---

## Step 7 — Initialise and Push to GitHub

Run these commands **one by one** in your terminal:

```bash
# Initialise a local git repository
git init

# Stage all files
git add .

# Create the first commit
git commit -m "Initial commit: AI-Powered Code Review and Bug Prediction System (IMSciences FYP 2021-2025)"

# Point to your GitHub repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ai-code-review.git

# Push to GitHub
git branch -M main
git push -u origin main
```

When prompted, enter your GitHub **username** and **password** (or Personal Access Token — see below).

---

## Step 8 — Personal Access Token (PAT) — Required for Password Authentication

GitHub no longer accepts plain passwords for `git push`. You need a PAT:

1. GitHub → **Settings** (click your avatar, top-right)
2. → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
3. → **Generate new token (classic)**
4. Give it a name, set expiry (e.g. 90 days), tick **repo** scope
5. Click **Generate token** — **copy it immediately** (it won't show again)
6. Use this token as your password when `git push` asks for it

---

## Step 9 — Add the Thesis PDF to the Repository

Copy your thesis PDF into the `docs/` folder before pushing:

```bash
# Windows (adjust path)
copy "C:\path\to\thesis.pdf" docs\Umair_Dost_FYP_Thesis.pdf

# Mac/Linux
cp /path/to/thesis.pdf docs/Umair_Dost_FYP_Thesis.pdf

git add docs/
git commit -m "Add thesis PDF to docs/"
git push
```

---

## Step 10 — Verify Your Repository

Visit:
```
https://github.com/YOUR_USERNAME/ai-code-review
```

You should see:
- ✅ The README with badges and architecture diagram rendering correctly
- ✅ All folders: `backend/`, `ml/`, `frontend/`, `configs/`, `docs/`, `.github/`
- ✅ The GitHub Actions CI tab (will show a pending/passing workflow)

---

## Step 11 — Polish for the MS Application (Recommended)

### Add a repository description and topics
On your repo page → click the ⚙️ gear icon (top-right of About section):
- **Description:** AI-Powered Code Review and Bug Prediction System — IMSciences BCS FYP 2025
- **Topics:** `machine-learning`, `bug-prediction`, `code-review`, `lstm`, `random-forest`, `fastapi`, `angular`, `python`
- **Website:** leave blank or add your university profile

### Pin the repository
GitHub profile → **Customize your profile** → **Pin repositories** → select `ai-code-review`

---

## Sharing the Link with the German University

Your repository URL to submit:
```
https://github.com/YOUR_USERNAME/ai-code-review
```

Make sure the repository is **Public** before submitting. The university
admissions team can browse the code, read the README, and view the full
project structure without needing a GitHub account.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `git: command not found` | Install Git (Step 3) |
| `remote: Repository not found` | Check the URL in Step 7 — must match your GitHub username exactly |
| `Authentication failed` | Use a Personal Access Token as password (Step 8) |
| Files too large | Add large model files (`.joblib`, `.h5`) to `.gitignore` — they are already excluded |
| Push rejected | Run `git pull origin main --rebase` first, then push again |
