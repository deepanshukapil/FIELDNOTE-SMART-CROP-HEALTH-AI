# GitHub par clean source snapshot upload

Yeh archive source files, actual datasets aur trained checkpoints rakhta hai, lekin purana `.git` commit history nahi. GitHub par yeh ek **naya import** hoga. GitHub commit uploader ki configured identity dikhayega; sirf upload karne se yeh prove nahi hota ki uploader ne har code line likhi hai. Apni identity aur attribution sahi rakhein.

## Upload steps

1. ZIP extract karke `smart-crop-health-ai-github-clean` folder open karein.
2. GitHub par **New repository** banayein; visibility apni zaroorat ke mutabik choose karein. GitHub ko README, license ya `.gitignore` auto-create na karne dein, kyunki files is folder mein pehle se hain.
3. Is folder mein terminal kholkar apni Git identity check karein:

   ```bash
   git config user.name
   git config user.email
   ```

   Agar values aapki sahi identity nahi hain, toh is repository ke liye apna actual GitHub name aur GitHub se verified email configure karein. GitHub ka private `noreply` email bhi use kar sakte hain.

4. Naye GitHub repo ka URL daal kar commands chalayein:

   ```bash
   git init
   git add .
   git commit -m "Initial source import"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   git push -u origin main
   ```

   `YOUR_USERNAME` aur `YOUR_REPOSITORY` ko apne values se replace karein.

## Is snapshot ki boundaries

- ZIP mein pehle ki Git history ya commit-author records shamil nahi hain.
- Yeh copy current live website se alag hai: GitHub par baad ke code changes apne-aap live site par deploy nahi honge.
- Public repository chunne par code, checkpoints aur bundled dataset files public honge; public karne se pehle dataset attribution/license notes `data/DATASETS.md` review karein.
