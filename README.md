# AI Learning Hub (Deep Learning Quiz App)

A modern, interactive web application for practicing Deep Learning and Machine Learning interview questions. Built with a focus on aesthetics and user experience.

## Features

- **50+ Curated Questions**: Covers Neural Networks, CNNs, RNNs, Transformers, and Diffusion Models.
- **Interactive Quiz Interface**: 
    - Smooth animations and glassmorphism design.
    - Instant feedback (Right/Wrong indicators).
    - Progress tracking.
- **Smart State Management**: 
    - Remembers your answers during the session.
    - Allows reviewing previous questions without revealing non-selected answers for retries (Practice Mode).
- **Responsive Design**: Fully optimized for Desktop and Mobile usage.

## project Structure

```
├── index.html      # Main application structure
├── style.css       # Modern styling with animations
├── script.js       # Core logic (State, Navigation, Grading)
├── questions.js    # Question data (Generated)
├── parser.py       # Utility to parse raw text to JSON
├── 题库.txt         # Original source data
└── README.md       # This file
```

## How to Run

### Method 1: Python Simple Server (Recommended)
Since this uses ES6 modules and fetches local data, it's best run on a local server.

1. Open your terminal in the project folder:
   ```bash
   cd path/to/PTA
   ```

2. Start the server:
   ```bash
   python3 -m http.server 8000
   ```

3. Open your browser:
   [http://localhost:8000](http://localhost:8000)

### Method 2: VS Code Live Server
If you use VS Code, simply right-click `index.html` and select "Open with Live Server".

## Development

- **Data Updates**: 
    1. Edit `题库.txt` to add new questions.
    2. Run `python3 parser.py` (you may need to adjust `parser.py` or `generate_js.py` depending on format changes) to regenerate `questions.js`.
- **Styling**: Modified via `style.css`. uses CSS Variables for easy theming.

## License
MIT
