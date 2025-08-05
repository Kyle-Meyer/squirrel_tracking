from src.downloader import ImageDownloader

# Use it with default paths (which will work from root)
tDownloader = ImageDownloader()
tDownloader.downloadAllImages()

# Debug script to see what's in downloader.py
import src.downloader

print("=== Contents of src.downloader module ===")
print("Available attributes:")
for item in dir(src.downloader):
    if not item.startswith('__'):
        print(f"  - {item}")

print("\n=== Looking for classes ===")
import inspect
for name, obj in inspect.getmembers(src.downloader):
    if inspect.isclass(obj):
        print(f"Found class: {name}")

print("\n=== Raw file contents (first 500 chars) ===")
with open('src/downloader.py', 'r') as f:
    content = f.read()
    print(content[:500])
    print(f"... (total length: {len(content)} characters)")
