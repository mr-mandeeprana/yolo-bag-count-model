import os
import shutil
import re
from pathlib import Path

def fix_dataset():
    root = Path(r"c:\Users\mrman\OneDrive\Desktop\Beumer work\Beumer Data\Yolo_bag_count_model")
    raw_dir = root / "data" / "raw" / "extracted_frames"
    processed_dir = root / "data" / "processed"
    
    splits = ["train", "val", "test"]
    
    print("--- Fixing Dataset Structure ---")
    
    for split in splits:
        img_dir = processed_dir / "images" / split
        lbl_dir = processed_dir / "labels" / split
        
        if not lbl_dir.exists():
            print(f"Skipping {split} - labels folder not found.")
            continue
            
        # 1. Clean up .txt files from images folder
        if img_dir.exists():
            txt_in_imgs = list(img_dir.glob("*.txt"))
            for txt_file in txt_in_imgs:
                # If it doesn't exist in labels yet, move it there? 
                # Better just to ensure it's in labels and remove from images.
                target_lbl = lbl_dir / txt_file.name
                if not target_lbl.exists():
                    shutil.move(str(txt_file), str(target_lbl))
                else:
                    txt_file.unlink()
        else:
            img_dir.mkdir(parents=True, exist_ok=True)

        # 2. Sync images based on labels
        labels = list(lbl_dir.glob("*.txt"))
        print(f"Processing {split}: found {len(labels)} labels.")
        
        for lbl_path in labels:
            base_name = lbl_path.stem
            
            # Simple heuristic to find original frame: 
            # Recording_2026-03-11_105738_f000000 - Copy (2) -> Recording_2026-03-11_105738_f000000
            original_frame_name = re.split(r' - Copy| \(', base_name)[0]
            src_img = raw_dir / f"{original_frame_name}.jpg"
            dst_img = img_dir / f"{base_name}.jpg"
            
            if src_img.exists():
                if not dst_img.exists():
                    shutil.copy(str(src_img), str(dst_img))
            else:
                print(f"  [WARN] Source image not found for {base_name} (looked for {src_img.name})")

    print("\n--- Done! ---")
    print("Total .txt files in processed:", len(list(processed_dir.rglob("*.txt"))))
    print("Total .jpg files in processed:", len(list(processed_dir.rglob("*.jpg"))))

if __name__ == "__main__":
    fix_dataset()
