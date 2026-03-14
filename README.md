This solution extracts the title and heading hierarchy (H1, H2, H3) from PDFs and outputs a structured JSON.


- Language: Python
- Library: PyMuPDF (`fitz`)

## How It Works
- Scans font sizes from text spans
- Determines top 3 sizes as H1 > H2 > H3
- Associates each heading with its page number

## How to Run

### Build
```bash
docker build --platform linux/amd64 -t mysolution:abc123 .
