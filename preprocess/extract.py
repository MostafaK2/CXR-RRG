import argparse
import zipfile
import io
import os
from pathlib import Path
from PIL import Image

def pad_to_square_and_save(img: Image.Image, out_path: Path, size: int = 512) -> None:
    img.thumbnail((size, size), Image.LANCZOS)  # shrink in place, keeps aspect ratio
    canvas = Image.new("RGB", (size, size), (0, 0, 0))  # black square
    offset_x = (size - img.width) // 2
    offset_y = (size - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    return canvas
    

def process_zip(zip_path: str): 
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = [e for e in zf.infolist() if not e.is_dir()]
        image_entries = [
            e for e in entries
            if Path(e.filename).suffix.lower() in ".png"
        ]
 
        total = len(image_entries)
        print(f"Found {total} image(s) in zip. Starting processing...\n")
        print(image_entries[0])
        
        ok, skipped, failed = 0, 0, 0
 
        for i, entry in enumerate(image_entries, 1):
            src_path = Path(entry.filename)         
            out_path = src_path.with_suffix(".png")

 
            # Skip if already processed
            if out_path.exists():
                print(f"[{i}/{total}] SKIP (exists)  {entry.filename}")
                skipped += 1
                continue
 
            try:
                with zf.open(entry) as f:
                    data = f.read()                        # read only this file into memory
                img = Image.open(io.BytesIO(data)).convert("RGB")
                pad_to_square_and_save(img, out_path)
                print(f"[{i}/{total}] OK            {entry.filename}  →  {out_path.name}")
                ok += 1
            except Exception as e:
                print(f"[{i}/{total}] FAIL          {entry.filename}  —  {e}")
                failed += 1
            
            
 
    print(f"\nDone. Processed: {ok}  |  Skipped: {skipped}  |  Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and process images from a zip file.")
    parser.add_argument("-z", help="Path to the input zip file containing images.")
    args = parser.parse_args()
    
    process_zip(args.z)