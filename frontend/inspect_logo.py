from PIL import Image
from collections import Counter

img = Image.open('src/assets/va-icon.png').convert('RGBA')
pixels = img.load()
width, height = img.size

colors = []
for y in range(height):
    for x in range(width):
        colors.append(pixels[x, y])

c = Counter(colors)
print("Most common colors:")
for color, count in c.most_common(10):
    print(f"Color: {color}, Count: {count}")
