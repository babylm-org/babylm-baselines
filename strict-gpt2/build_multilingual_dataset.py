"""
Build a multilingual BabyLM dataset by sampling from per-language sources.

Usage:
    python build_multilingual_dataset.py --languages en nld zho \
        --words-per-lang 50000000 --output data/multilingual-50M

Each language is sampled to just under (words_per_lang * byte_premium) words,
preserving the proportion of words from each data source within that language.

- English (.txt): preserves continuity by taking contiguous text from the start.
- NLD/ZHO (.parquet): keeps whole rows when sampling.

Output files are prefixed with the language code to avoid name conflicts.
"""

import argparse
from pathlib import Path

import pandas as pd


BYTE_PREMIUMS = {
    'en': 1.0,
    'nld': 1.0516,
    'zho': 0.935966,
}

LANGUAGE_DATASETS = {
    'en': 'BabyLM-2026-Strict',
    'nld': 'babylm-nld',
    'zho': 'babylm-zho',
}


def count_words_text(text: str) -> int:
    return len(text.split())


def load_english_sources(data_dir: Path) -> dict[str, str]:
    """Load English .txt sources, returning {source_name: text}."""
    sources = {}
    for f in sorted(data_dir.glob('*.train.txt')):
        source_name = f.name.replace('.train.txt', '')
        sources[source_name] = f.read_text()
    return sources


def load_parquet_sources(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load parquet sources, returning {source_name: dataframe}."""
    sources = {}
    for f in sorted(data_dir.glob('*.train.parquet')):
        source_name = f.name.replace('.train.parquet', '')
        sources[source_name] = pd.read_parquet(f)
    return sources


def sample_english_source(text: str, target_words: int) -> str:
    """Take a contiguous chunk from the start, truncating at a word boundary
    to land just under target_words."""
    words = text.split()
    if len(words) <= target_words:
        return text
    # Rejoin the first target_words words
    # Find the byte offset of the end of the target_words-th word
    # to preserve original whitespace/newlines
    count = 0
    end_pos = 0
    for match in __import__('re').finditer(r'\S+', text):
        count += 1
        if count >= target_words:
            end_pos = match.end()
            break
    return text[:end_pos]


def sample_parquet_source(df: pd.DataFrame, target_words: int) -> pd.DataFrame:
    """Take whole rows from the start until just under target_words (using num-tokens)."""
    cumulative = df['num-tokens'].cumsum()
    # Find the last row where cumulative sum is still under target
    mask = cumulative <= target_words
    if not mask.any():
        # Even the first row exceeds target — take nothing
        return df.iloc[:0]
    last_idx = mask.values.nonzero()[0][-1]
    return df.iloc[:last_idx + 1]


def build_language(lang: str, target_words: int, data_root: Path, output_dir: Path):
    adjusted_target = int(target_words * BYTE_PREMIUMS[lang])
    dataset_dir = data_root / LANGUAGE_DATASETS[lang]
    is_english = (lang == 'en')

    print(f'\n  [{lang}] target: {target_words:,} words, '
          f'adjusted (x{BYTE_PREMIUMS[lang]}): {adjusted_target:,} words')

    if is_english:
        sources = load_english_sources(dataset_dir)
        # Count words per source
        source_word_counts = {name: count_words_text(text) for name, text in sources.items()}
    else:
        sources = load_parquet_sources(dataset_dir)
        # Use num-tokens column for word counts
        source_word_counts = {name: int(df['num-tokens'].sum()) for name, df in sources.items()}

    total_words = sum(source_word_counts.values())
    print(f'  [{lang}] total available: {total_words:,} words across {len(sources)} sources')

    if adjusted_target >= total_words:
        print(f'  [{lang}] WARNING: target ({adjusted_target:,}) >= available ({total_words:,}), '
              f'using all data')

    # Calculate per-source targets preserving proportions
    source_targets = {}
    for name, wc in source_word_counts.items():
        proportion = wc / total_words
        source_targets[name] = int(adjusted_target * proportion)

    # Sample and save each source
    total_sampled = 0
    for name in sorted(sources.keys()):
        target = source_targets[name]
        out_name_prefix = f'{lang}_{name}'

        if is_english:
            sampled_text = sample_english_source(sources[name], target)
            sampled_words = count_words_text(sampled_text)
            out_path = output_dir / f'{out_name_prefix}.train.txt'
            out_path.write_text(sampled_text)
        else:
            sampled_df = sample_parquet_source(sources[name], target)
            sampled_words = int(sampled_df['num-tokens'].sum()) if len(sampled_df) > 0 else 0
            out_path = output_dir / f'{out_name_prefix}.train.parquet'
            sampled_df.to_parquet(out_path, index=False)

        total_sampled += sampled_words
        available = source_word_counts[name]
        print(f'    {name}: {sampled_words:,} / {available:,} words '
              f'(target {target:,}) -> {out_path.name}')

    print(f'  [{lang}] total sampled: {total_sampled:,} / adjusted target {adjusted_target:,}')


def main():
    parser = argparse.ArgumentParser(
        description='Build a multilingual BabyLM dataset.')
    parser.add_argument('--languages', nargs='+', required=True,
                        choices=list(LANGUAGE_DATASETS.keys()),
                        help='Languages to include (en, nld, zho)')
    parser.add_argument('--words-per-lang', type=int, required=True,
                        help='Target number of words per language (before byte premium)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (e.g. data/multilingual-50M)')
    parser.add_argument('--data-root', type=str, default='data',
                        help='Root data directory (default: data)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    print(f'Building multilingual dataset: {args.languages}')
    print(f'Target: {args.words_per_lang:,} words/lang -> {args.output}')

    for lang in args.languages:
        build_language(lang, args.words_per_lang, data_root, output_dir)

    print(f'\nDone. Output written to {output_dir}/')


if __name__ == '__main__':
    main()
