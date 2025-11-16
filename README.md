# chess-accuracy-tools
Study relationship between chess playing accuracy scores (like those on Chess.com or Lichess) and Elo strength (online or over-the-board).

# ♟️ Chess Accuracy and Strength Analysis

This repository is dedicated to investigating the relationship between calculated **chess engine accuracy scores** (e.g., Stockfish evaluations) and established **player strength measures** (e.g., Elo rating, FIDE/USCF titles).

It serves as a central hub for the source code, analysis notes, and collated data necessary for this research.

---

## 📁 Repository Structure

The project is organized into four main directories to separate logic from assets:

| Directory | Content | Licensing Note |
| :--- | :--- | :--- |
| **`src/`** | Python scripts for data processing, accuracy score calculation, and statistical analysis. | MIT License |
| **`data/`** | Collated CSV files containing game records and raw data for analysis. | CC BY 4.0 |
| **`binaries/`** | Executable files, including the **Stockfish engine binary** used for generating accuracy scores. | N/A (External License) |
| **`notes/`** | Plain text logs, research findings, and documentation notes. | CC BY 4.0 |

---

## ⚙️ Requirements & Setup

To use the code and contribute to this project, you will need the following tools installed on your system:

### 1. Git and Git LFS

This repository uses **Git Large File Storage (Git LFS)** to manage the large CSV data files and the Stockfish binary efficiently.

```bash
# Install Git LFS (needed on both Windows and Debian)
git lfs install

# Clone the repository
git clone https://github.com/ZAF-prog/chess-accuracy-tools/
