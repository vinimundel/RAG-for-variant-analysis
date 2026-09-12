"""Ask a literature question using a local hybrid index."""
import argparse
from pathlib import Path
from healthrag.rag.biomedical import BiomedicalRAG, Question
from healthrag.rag.profiles import ResearchProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', required=True)
    parser.add_argument('--collection', default='BIOMEDICAL')
    parser.add_argument('--profile', default='general', choices=['general', 'braf_oncology'])
    parser.add_argument('--profile-file', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    profile = ResearchProfile.model_validate_json(args.profile_file.read_text()) if args.profile_file else None
    result = BiomedicalRAG(args.workspace, profile).invoke(Question(
        question=args.question, collection=args.collection, profile=args.profile, k=args.k))
    text = result.model_dump_json(indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)
    if result.status in {'invalid_generation', 'generation_failed'}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
