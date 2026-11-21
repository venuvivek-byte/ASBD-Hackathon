"""
Creates the final hackathon submission package: outputs/ASBD_Hackathon_Submission.zip
Includes:
  - data/
  - src/
  - notebooks/
  - outputs/plots/
  - outputs/rules/
  - report/
  - README.md

Author: ASBD Hackathon Team
"""

import os
import zipfile

def create_submission_zip():
    zip_path = os.path.join("outputs", "ASBD_Hackathon_Submission.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print(f"Creating submission archive: {zip_path}...")
    
    include_dirs = ["data", "src", "notebooks", "report", os.path.join("outputs", "plots"), os.path.join("outputs", "rules")]
    include_files = ["README.md"]
    
    file_count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add root files
        for f in include_files:
            if os.path.exists(f):
                zipf.write(f, arcname=f)
                file_count += 1
                print(f"  + Added: {f}")
                
        # Add directories
        for d in include_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, files in os.walk(d):
                # Exclude cache directories
                dirs[:] = [sub for sub in dirs if sub not in ['__pycache__', '.ipynb_checkpoints']]
                for file in files:
                    # Skip existing submission zip if inside outputs
                    if file == "ASBD_Hackathon_Submission.zip" or file.endswith(".pyc"):
                        continue
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, ".")
                    zipf.write(full_path, arcname=rel_path)
                    file_count += 1
                    
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n[OK] Submission Package successfully created: {zip_path}")
    print(f"     Total files included: {file_count}")
    print(f"     Archive file size: {zip_size_mb:.2f} MB")

if __name__ == "__main__":
    create_submission_zip()
