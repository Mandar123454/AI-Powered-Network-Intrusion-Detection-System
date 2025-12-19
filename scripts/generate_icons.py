"""
Generate PWA Icons and Social Preview Image for AI-NIDS
========================================================
Run this script to generate all required icons for PWA support.

Requirements:
    pip install pillow cairosvg

Usage:
    python scripts/generate_icons.py
"""

import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Installing Pillow...")
    os.system("pip install pillow")
    from PIL import Image, ImageDraw, ImageFont

# Icon sizes for PWA
ICON_SIZES = [72, 96, 128, 144, 152, 192, 384, 512]

# Colors
BG_COLOR = (10, 15, 26)  # #0a0f1a - Dark blue
PRIMARY_COLOR = (59, 130, 246)  # #3b82f6 - Blue
ACCENT_COLOR = (0, 240, 255)  # Cyan accent
WHITE = (255, 255, 255)
SUCCESS_COLOR = (16, 185, 129)  # Green


def create_icon(size, output_path):
    """Create a single PWA icon."""
    # Create image with dark background
    img = Image.new('RGBA', (size, size), BG_COLOR + (255,))
    draw = ImageDraw.Draw(img)
    
    # Calculate shield dimensions
    padding = size * 0.1
    shield_width = size - (padding * 2)
    shield_height = shield_width * 1.2
    
    # Center position
    cx, cy = size // 2, size // 2
    
    # Draw shield outline (gradient-like effect)
    for i in range(3):
        offset = i * 2
        draw.polygon([
            (cx, cy - shield_height//2 + offset),  # Top
            (cx + shield_width//2 - offset, cy - shield_height//4),  # Top right
            (cx + shield_width//2 - offset, cy + shield_height//4),  # Bottom right
            (cx, cy + shield_height//2 - offset),  # Bottom
            (cx - shield_width//2 + offset, cy + shield_height//4),  # Bottom left
            (cx - shield_width//2 + offset, cy - shield_height//4),  # Top left
        ], outline=(*PRIMARY_COLOR, 255 - i*50), width=max(1, size//30))
    
    # Draw checkmark inside
    check_size = shield_width * 0.4
    check_x = cx - check_size * 0.3
    check_y = cy
    
    # Checkmark path
    draw.line([
        (check_x - check_size*0.2, check_y),
        (check_x, check_y + check_size*0.3),
        (check_x + check_size*0.4, check_y - check_size*0.3)
    ], fill=SUCCESS_COLOR + (255,), width=max(2, size//25))
    
    # Save with proper format
    img.save(output_path, 'PNG', optimize=True)
    print(f"  ✓ Created {size}x{size} icon")


def create_social_preview(output_path, width=1280, height=640):
    """Create social preview image for GitHub/Twitter."""
    img = Image.new('RGB', (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Create gradient background effect
    for y in range(height):
        # Subtle gradient
        factor = y / height
        r = int(BG_COLOR[0] + (20 - BG_COLOR[0]) * factor * 0.3)
        g = int(BG_COLOR[1] + (30 - BG_COLOR[1]) * factor * 0.3)
        b = int(BG_COLOR[2] + (50 - BG_COLOR[2]) * factor * 0.3)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Draw grid pattern (cyber effect)
    grid_color = (30, 40, 60)
    for x in range(0, width, 40):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, 40):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    # Draw main shield icon (left side)
    shield_x, shield_y = 200, height // 2
    shield_size = 200
    
    # Shield polygon
    points = [
        (shield_x, shield_y - shield_size//2),
        (shield_x + shield_size//2, shield_y - shield_size//3),
        (shield_x + shield_size//2, shield_y + shield_size//3),
        (shield_x, shield_y + shield_size//2),
        (shield_x - shield_size//2, shield_y + shield_size//3),
        (shield_x - shield_size//2, shield_y - shield_size//3),
    ]
    
    # Draw shield with glow
    for i in range(5, 0, -1):
        glow_color = (PRIMARY_COLOR[0]//i, PRIMARY_COLOR[1]//i, PRIMARY_COLOR[2]//i)
        draw.polygon(points, outline=glow_color, width=i*3)
    
    draw.polygon(points, outline=PRIMARY_COLOR, width=4)
    
    # Checkmark inside shield
    check_points = [
        (shield_x - 40, shield_y),
        (shield_x - 10, shield_y + 40),
        (shield_x + 50, shield_y - 40)
    ]
    draw.line(check_points, fill=SUCCESS_COLOR, width=8)
    
    # Try to use a font, fallback to default
    try:
        title_font = ImageFont.truetype("arial.ttf", 72)
        subtitle_font = ImageFont.truetype("arial.ttf", 32)
        desc_font = ImageFont.truetype("arial.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        desc_font = ImageFont.load_default()
    
    # Title text
    title = "AI-NIDS"
    draw.text((450, 180), title, fill=WHITE, font=title_font)
    
    # Subtitle
    subtitle = "AI-Powered Network Intrusion Detection System"
    draw.text((450, 280), subtitle, fill=PRIMARY_COLOR, font=subtitle_font)
    
    # Description
    desc_lines = [
        "• Real-time Threat Detection with ML/AI",
        "• SHAP Explainability & MITRE ATT&CK Mapping",
        "• SOC-Grade Security Analytics Dashboard",
    ]
    
    y_pos = 360
    for line in desc_lines:
        draw.text((450, y_pos), line, fill=(180, 180, 180), font=desc_font)
        y_pos += 40
    
    # Bottom accent line
    draw.line([(0, height - 5), (width, height - 5)], fill=PRIMARY_COLOR, width=5)
    
    # Tech stack badges area
    badges_y = height - 80
    badge_texts = ["Python", "Flask", "TensorFlow", "Chart.js", "Bootstrap"]
    badge_x = 450
    
    for badge in badge_texts:
        # Badge background
        badge_width = len(badge) * 12 + 20
        draw.rounded_rectangle(
            [badge_x, badges_y, badge_x + badge_width, badges_y + 30],
            radius=5,
            fill=(40, 50, 70),
            outline=(60, 80, 100)
        )
        draw.text((badge_x + 10, badges_y + 5), badge, fill=(150, 160, 180), font=desc_font)
        badge_x += badge_width + 15
    
    img.save(output_path, 'PNG', quality=95, optimize=True)
    print(f"  ✓ Created social preview ({width}x{height})")


def create_favicon(output_path):
    """Create favicon.ico with multiple sizes."""
    sizes = [16, 32, 48]
    images = []
    
    for size in sizes:
        img = Image.new('RGBA', (size, size), BG_COLOR + (255,))
        draw = ImageDraw.Draw(img)
        
        # Simple shield shape
        cx, cy = size // 2, size // 2
        s = size * 0.4
        
        points = [
            (cx, cy - s),
            (cx + s, cy - s*0.5),
            (cx + s, cy + s*0.5),
            (cx, cy + s),
            (cx - s, cy + s*0.5),
            (cx - s, cy - s*0.5),
        ]
        
        draw.polygon(points, fill=PRIMARY_COLOR + (255,))
        
        # Checkmark
        draw.line([
            (cx - s*0.3, cy),
            (cx, cy + s*0.3),
            (cx + s*0.4, cy - s*0.3)
        ], fill=WHITE + (255,), width=max(1, size//10))
        
        images.append(img)
    
    # Save as ICO
    images[0].save(
        output_path,
        format='ICO',
        sizes=[(s, s) for s in sizes],
        append_images=images[1:]
    )
    print(f"  ✓ Created favicon.ico")


def main():
    """Generate all icons and images."""
    print("\n🎨 Generating AI-NIDS Icons and Images...\n")
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    static_dir = project_root / 'app' / 'static'
    images_dir = static_dir / 'images'
    icons_dir = images_dir / 'icons'
    
    # Create directories
    icons_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / 'screenshots').mkdir(exist_ok=True)
    
    print("📱 Creating PWA icons...")
    for size in ICON_SIZES:
        output_path = icons_dir / f'icon-{size}x{size}.png'
        create_icon(size, output_path)
    
    print("\n🖼️  Creating social preview...")
    create_social_preview(images_dir / 'social-preview.png')
    
    print("\n⭐ Creating favicon...")
    create_favicon(static_dir / 'favicon.ico')
    
    print("\n✅ All icons generated successfully!")
    print(f"   📁 Output: {images_dir}")
    
    # Print usage instructions
    print("\n📋 Next steps:")
    print("   1. Add to base.html <head>:")
    print('      <link rel="manifest" href="/static/manifest.json">')
    print('      <link rel="icon" href="/static/favicon.ico">')
    print('      <meta name="theme-color" content="#3b82f6">')
    print("   2. Register service worker in your main JS:")
    print("      if ('serviceWorker' in navigator) {")
    print("          navigator.serviceWorker.register('/static/sw.js');")
    print("      }")


if __name__ == '__main__':
    main()
