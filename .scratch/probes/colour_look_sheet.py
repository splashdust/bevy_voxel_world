"""Throwaway: contact sheets of the coloured-light look probe's shots.

Usage: python3 colour_look_sheet.py <shots dir> <out dir>
Each sheet lays named shots in a grid, each labelled with its file name.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

SHEETS = {
    "1_cave": ["cave_0_today", "cave_1_component", "cave_2_max", "cave_3_sum", "cave_4_blend", "cave_5_blendlevel"],
    "2_cave_bright": ["bright_cave_1_component", "bright_cave_2_max", "bright_cave_3_sum", "bright_cave_4_blend"],
    "3_overhang": ["overhang_0_today", "overhang_1_component", "overhang_2_max", "overhang_3_sum"],
    "4_island": ["island_0_today", "island_1_component", "island_2_max", "island_3_sum"],
    "5_stages": ["check_stages", "stages_1_component", "stages_2_max", "stages_3_sum",
                 "stages_4_blend", "check_cave"],
    "6_trenches": ["trenches_0_today", "trenches_1_component", "trenches_2_max",
                   "light_trenches_1_component", "light_trenches_2_max", "bright_trenches_1_component"],
    "7_glow": ["cave_2_max_glow_material", "cave_2_max", "cave_2_max_glow_both",
               "stages_2_max_glow_material", "stages_2_max", "stages_2_max_glow_both"],
    "8_day": ["day_cave_2_max", "day_overhang_1_component", "day_overhang_2_max",
              "day_island_1_component", "day_island_2_max", "day_stages_2_max"],
    "9_linear": ["linear_trenches_0_today", "linear_trenches_1_component", "linear_trenches_2_max",
                 "linear_cave_1_component", "linear_cave_2_max", "linear_cave_3_sum",
                 "linear_cave_4_blend", "linear_cave_5_blendlevel", "overhang_4_blend"],
    "10_reach32": ["trenches_1_component", "r32_trenches_1_component", "r32_256_trenches_1_component",
                   "trenches_2_max", "r32_trenches_2_max", "r32_256_trenches_2_max",
                   "cave_2_max", "r32_cave_2_max", "r32_256_cave_2_max"],
}

WIDTH = 700


def main():
    shots, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    for sheet, names in SHEETS.items():
        names = [n for n in names if os.path.exists(f"{shots}/{n}.png")]
        if not names:
            continue
        images = [Image.open(f"{shots}/{n}.png").convert("RGB") for n in names]
        w = WIDTH
        h = int(images[0].height * w / images[0].width)
        cols = 2 if len(images) <= 4 else (3 if len(images) in (6, 9) else 4)
        rows = (len(images) + cols - 1) // cols
        page = Image.new("RGB", (cols * w, rows * h), "black")
        draw = ImageDraw.Draw(page)
        for i, (name, image) in enumerate(zip(names, images)):
            x, y = (i % cols) * w, (i // cols) * h
            page.paste(image.resize((w, h)), (x, y))
            draw.rectangle([x, y, x + 12 * len(name) + 12, y + 30], fill="black")
            draw.text((x + 6, y + 3), name, fill="white", font=font)
        page.save(f"{out}/{sheet}.png")
        print(f"{out}/{sheet}.png")


main()
