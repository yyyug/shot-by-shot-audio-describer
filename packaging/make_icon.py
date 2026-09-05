"""Generate packaging/shot.ico (a simple 32x32 app icon) without PIL."""
import struct
import os

SIZE = 32
NAVY = (28, 40, 68, 255)
ORANGE = (255, 165, 0, 255)
WHITE = (255, 255, 255, 255)


def pixel(x, y):
    # Rounded-square navy background
    cx = (x + 0.5) - SIZE / 2.0
    cy = (y + 0.5) - SIZE / 2.0
    r = SIZE / 2.0
    if (abs(cx) > r - 2 or abs(cy) > r - 2) or (cx * cx + cy * cy) > (r - 1) ** 2:
        return (0, 0, 0, 0)
    # Play triangle (left of center, pointing right)
    px = (x + 0.5) / SIZE
    py = (y + 0.5) / SIZE
    if 0.34 <= px <= 0.62 and abs((py - 0.5) * 1.6) < (px - 0.34):
        return ORANGE
    # "A" hint via a white bar under the triangle
    if 0.30 <= px <= 0.70 and 0.78 <= py <= 0.84:
        return WHITE
    return NAVY


def build_xor_rows():
    rows = []
    for y in range(SIZE - 1, -1, -1):  # bottom-up
        row = bytearray()
        for x in range(SIZE):
            b, g, r, a = pixel(x, y)
            row += struct.pack("BBBB", b, g, r, a)
        rows.append(bytes(row))
    return b"".join(rows)


def build_and_mask():
    rows = []
    for y in range(SIZE - 1, -1, -1):
        row = bytearray()
        for x in range(0, SIZE, 8):
            byte = 0
            for bit in range(8):
                b, g, r, a = pixel(x + bit, y)
                if a == 0:
                    byte |= 1 << (7 - bit)
            row.append(byte)
        # pad to 4-byte boundary (32px -> exactly 4 bytes, no padding needed)
        while len(row) % 4:
            row.append(0)
        rows.append(bytes(row))
    return b"".join(rows)


def main():
    bmp_header = struct.pack(
        "<IIIHHIIIIII",
        40, SIZE, SIZE * 2, 1, 32, 0, SIZE * SIZE * 4, 0, 0, 0, 0,
    )
    image_data = bmp_header + build_xor_rows() + build_and_mask()

    icon_dir = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack(
        "<BBBBHHII",
        SIZE, SIZE, 0, 0, 1, 32, len(image_data), 22,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shot.ico")
    with open(out_path, "wb") as f:
        f.write(icon_dir + entry + image_data)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
