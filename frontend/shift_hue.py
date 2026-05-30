from PIL import Image

def colorize_to_gold_and_charcoal(input_path, output_path):
    img = Image.open(input_path).convert('RGBA')
    pixels = img.load()
    width, height = img.size
    
    # Gold: #C6A75E -> RGB(198, 167, 94)
    # Charcoal: #151412 -> RGB(21, 20, 18)
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            
            # If the pixel is fully transparent, keep it transparent
            if a == 0:
                continue
                
            brightness = (r + g + b) / 3.0
            
            # If it's a dark background pixel, change to Charcoal
            if brightness < 35:
                pixels[x, y] = (21, 20, 18, a)
            else:
                # Anti-aliased blend factor between Charcoal and Gold based on brightness
                # Lowering the upper limit to 110.0 to make the Gold wave extremely bright and saturated
                factor = (brightness - 35) / (110.0 - 35)
                factor = max(0.0, min(1.0, factor))
                
                r_new = int(21 + (198 - 21) * factor)
                g_new = int(20 + (167 - 20) * factor)
                b_new = int(18 + (94 - 18) * factor)
                
                pixels[x, y] = (r_new, g_new, b_new, a)

    img.save(output_path)
    print(f"Successfully colorized {input_path} to {output_path} (Charcoal & Bright Gold)")

if __name__ == "__main__":
    colorize_to_gold_and_charcoal("src/assets/va-icon.png", "public/va-icon.png")
    colorize_to_gold_and_charcoal("src/assets/va-icon.png", "src/assets/va-icon.png")
