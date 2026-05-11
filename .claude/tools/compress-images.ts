import { tool } from "@anthropic-ai/tool";
import { z } from "zod";

export default tool({
  description:
    "Compress images (JPEG/PNG) in a directory to reduce file size. Uses Pillow to re-encode images with optimized settings. Reports before/after sizes for each file.",
  args: z.object({
    path: z
      .string()
      .default("examples")
      .describe(
        "File or directory path to compress images in (default: examples)"
      ),
    quality: z
      .number()
      .min(1)
      .max(100)
      .default(85)
      .describe("JPEG quality level 1-100 (default: 85). Lower = smaller file"),
    max_width: z
      .number()
      .optional()
      .describe(
        "Optional max width in pixels. Images wider than this will be resized proportionally"
      ),
    dry_run: z
      .boolean()
      .default(false)
      .describe("If true, only report what would be done without modifying files"),
  }),
  async execute({ path, quality, max_width, dry_run }) {
    const script = `
import os
import sys
from pathlib import Path
from PIL import Image

target = Path("${path}")
quality = ${quality}
max_width = ${max_width ?? "None"}
dry_run = ${dry_run ? "True" : "False"}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

def find_images(p: Path):
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
        return [p]
    elif p.is_dir():
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(p.rglob(f"*{ext}"))
        return sorted(images)
    return []

images = find_images(target)
if not images:
    print(f"No images found in {target}")
    sys.exit(0)

total_before = 0
total_after = 0
results = []

for img_path in images:
    before_size = img_path.stat().st_size
    total_before += before_size

    if dry_run:
        results.append(f"  [DRY RUN] {img_path} ({before_size / 1024:.1f}K)")
        continue

    try:
        img = Image.open(img_path)

        # Resize if max_width specified
        if max_width and img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)

        # Determine save parameters based on format
        suffix = img_path.suffix.lower()
        save_kwargs = {}

        if suffix in ('.jpg', '.jpeg'):
            save_kwargs = {'quality': quality, 'optimize': True}
            # Preserve EXIF if present
            if 'exif' in img.info:
                save_kwargs['exif'] = img.info['exif']
        elif suffix == '.png':
            save_kwargs = {'optimize': True}
            # Convert RGBA to RGB if no transparency needed
        elif suffix == '.webp':
            save_kwargs = {'quality': quality, 'method': 6}

        img.save(img_path, **save_kwargs)
        after_size = img_path.stat().st_size
        total_after += after_size
        reduction = (1 - after_size / before_size) * 100 if before_size > 0 else 0
        results.append(f"  {img_path}: {before_size / 1024:.1f}K -> {after_size / 1024:.1f}K ({reduction:.1f}% reduction)")
    except Exception as e:
        results.append(f"  ERROR {img_path}: {e}")
        total_after += before_size

print(f"Images found: {len(images)}")
print(f"Settings: quality={quality}, max_width={max_width or 'none'}, dry_run={dry_run}")
print()
for r in results:
    print(r)

if not dry_run:
    total_reduction = (1 - total_after / total_before) * 100 if total_before > 0 else 0
    print()
    print(f"Total: {total_before / 1024:.1f}K -> {total_after / 1024:.1f}K ({total_reduction:.1f}% reduction)")
`;

    const result = Bun.spawnSync(["uv", "run", "python", "-c", script], {
      cwd: import.meta.dir + "/../..",
    });
    const stdout = result.stdout.toString();
    const stderr = result.stderr.toString();
    return `Exit code: ${result.exitCode}\n${stdout}\n${stderr}`.trim();
  },
});
