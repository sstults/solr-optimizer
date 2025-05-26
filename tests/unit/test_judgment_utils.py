"""
Unit tests for judgment utilities.
"""

import pytest
import tempfile
import csv
import json
from pathlib import Path

from solr_optimizer.utils.judgment_utils import (
    JudgmentEntry, JudgmentLoader, JudgmentSaver, JudgmentValidator,
    load_judgments, save_judgments
)


class TestJudgmentEntry:
    """Test cases for the JudgmentEntry class."""

    def test_judgment_entry_creation(self):
        """Test creating a judgment entry."""
        entry = JudgmentEntry(
            query="laptop",
            document_id="doc1",
            relevance=3,
            metadata={"source": "manual"}
        )
        
        assert entry.query == "laptop"
        assert entry.document_id == "doc1"
        assert entry.relevance == 3
        assert entry.metadata == {"source": "manual"}

    def test_judgment_entry_minimal(self):
        """Test creating judgment entry with minimal data."""
        entry = JudgmentEntry(
            query="smartphone",
            document_id="doc2",
            relevance=1
        )
        
        assert entry.query == "smartphone"
        assert entry.document_id == "doc2"
        assert entry.relevance == 1
        assert entry.metadata is None


class TestJudgmentLoader:
    """Test cases for the JudgmentLoader class."""

    def test_load_from_csv_standard_format(self):
        """Test loading judgments from standard CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "judgments.csv"
            
            # Create test CSV
            csv_data = [
                ["query", "document_id", "relevance_score"],
                ["laptop", "doc1", "3"],
                ["laptop", "doc2", "1"],
                ["smartphone", "doc3", "2"],
                ["smartphone", "doc4", "3"],
                ["tablet", "doc5", "0"]
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            loader = JudgmentLoader()
            judgments = loader.load_from_csv(csv_file)
            
            assert len(judgments) == 3
            assert "laptop" in judgments
            assert "smartphone" in judgments
            assert "tablet" in judgments
            
            assert judgments["laptop"]["doc1"] == 3
            assert judgments["laptop"]["doc2"] == 1
            assert judgments["smartphone"]["doc3"] == 2
            assert judgments["smartphone"]["doc4"] == 3
            assert judgments["tablet"]["doc5"] == 0

    def test_load_from_csv_custom_columns(self):
        """Test loading judgments from CSV with custom column names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "custom_judgments.csv"
            
            # Create test CSV with custom column names
            csv_data = [
                ["search_query", "doc_identifier", "score"],
                ["laptop", "doc1", "3"],
                ["smartphone", "doc2", "2"]
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            loader = JudgmentLoader()
            judgments = loader.load_from_csv(
                csv_file,
                query_col="search_query",
                doc_col="doc_identifier",
                relevance_col="score"
            )
            
            assert len(judgments) == 2
            assert judgments["laptop"]["doc1"] == 3
            assert judgments["smartphone"]["doc2"] == 2

    def test_load_from_csv_missing_columns(self):
        """Test loading from CSV with missing required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "invalid_judgments.csv"
            
            # Create CSV missing relevance_score column
            csv_data = [
                ["query", "document_id"],
                ["laptop", "doc1"]
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            loader = JudgmentLoader()
            
            with pytest.raises(ValueError, match="CSV must contain columns"):
                loader.load_from_csv(csv_file)

    def test_load_from_csv_invalid_relevance_scores(self):
        """Test loading from CSV with invalid relevance scores."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "invalid_scores.csv"
            
            # Create CSV with non-integer relevance scores
            csv_data = [
                ["query", "document_id", "relevance_score"],
                ["laptop", "doc1", "3"],
                ["laptop", "doc2", "invalid"],  # Invalid score
                ["smartphone", "doc3", "2"]
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            loader = JudgmentLoader()
            judgments = loader.load_from_csv(csv_file)
            
            # Should skip invalid row and continue
            assert len(judgments) == 2
            assert judgments["laptop"]["doc1"] == 3
            assert judgments["smartphone"]["doc3"] == 2
            assert "doc2" not in judgments["laptop"]

    def test_load_from_csv_empty_values(self):
        """Test loading from CSV with empty query/doc values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "empty_values.csv"
            
            csv_data = [
                ["query", "document_id", "relevance_score"],
                ["laptop", "doc1", "3"],
                ["", "doc2", "1"],  # Empty query
                ["smartphone", "", "2"],  # Empty doc_id
                ["tablet", "doc3", "0"]
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            loader = JudgmentLoader()
            judgments = loader.load_from_csv(csv_file)
            
            # Should skip rows with empty values
            assert len(judgments) == 2
            assert "laptop" in judgments
            assert "tablet" in judgments

    def test_load_from_trec_qrels(self):
        """Test loading judgments from TREC qrels format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            qrels_file = Path(temp_dir) / "qrels.txt"
            
            # Create TREC qrels file
            with open(qrels_file, 'w') as f:
                f.write("1 0 doc1 3\n")
                f.write("1 0 doc2 1\n")
                f.write("2 0 doc3 2\n")
                f.write("2 0 doc4 3\n")
                f.write("3 0 doc5 0\n")
            
            loader = JudgmentLoader()
            judgments = loader.load_from_trec_qrels(qrels_file)
            
            assert len(judgments) == 3
            assert "1" in judgments
            assert "2" in judgments
            assert "3" in judgments
            
            assert judgments["1"]["doc1"] == 3
            assert judgments["1"]["doc2"] == 1
            assert judgments["2"]["doc3"] == 2
            assert judgments["2"]["doc4"] == 3
            assert judgments["3"]["doc5"] == 0

    def test_load_from_trec_qrels_with_comments(self):
        """Test loading TREC qrels with comments and empty lines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            qrels_file = Path(temp_dir) / "qrels_with_comments.txt"
            
            with open(qrels_file, 'w') as f:
                f.write("# This is a comment\n")
                f.write("1 0 doc1 3\n")
                f.write("\n")  # Empty line
                f.write("2 0 doc2 2\n")
                f.write("# Another comment\n")
                f.write("3 0 doc3 1\n")
            
            loader = JudgmentLoader()
            judgments = loader.load_from_trec_qrels(qrels_file)
            
            assert len(judgments) == 3
            assert judgments["1"]["doc1"] == 3
            assert judgments["2"]["doc2"] == 2
            assert judgments["3"]["doc3"] == 1

    def test_load_from_trec_qrels_invalid_format(self):
        """Test loading TREC qrels with invalid format lines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            qrels_file = Path(temp_dir) / "invalid_qrels.txt"
            
            with open(qrels_file, 'w') as f:
                f.write("1 0 doc1 3\n")
                f.write("invalid line\n")  # Invalid format
                f.write("2 0 doc2 2\n")
                f.write("incomplete\n")  # Another invalid line
                f.write("3 0 doc3 1\n")
            
            loader = JudgmentLoader()
            judgments = loader.load_from_trec_qrels(qrels_file)
            
            # Should skip invalid lines
            assert len(judgments) == 3
            assert judgments["1"]["doc1"] == 3
            assert judgments["2"]["doc2"] == 2
            assert judgments["3"]["doc3"] == 1

    def test_load_from_json(self):
        """Test loading judgments from JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "judgments.json"
            
            # Create test JSON
            data = {
                "laptop": {"doc1": 3, "doc2": 1, "doc3": 0},
                "smartphone": {"doc4": 2, "doc5": 3},
                "tablet": {"doc6": 1}
            }
            
            with open(json_file, 'w') as f:
                json.dump(data, f)
            
            loader = JudgmentLoader()
            judgments = loader.load_from_json(json_file)
            
            assert len(judgments) == 3
            assert judgments["laptop"]["doc1"] == 3
            assert judgments["laptop"]["doc2"] == 1
            assert judgments["smartphone"]["doc4"] == 2
            assert judgments["tablet"]["doc6"] == 1

    def test_load_from_json_invalid_structure(self):
        """Test loading from JSON with invalid structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "invalid_judgments.json"
            
            # Create JSON that's not a dictionary at root
            data = ["not", "a", "dictionary"]
            
            with open(json_file, 'w') as f:
                json.dump(data, f)
            
            loader = JudgmentLoader()
            
            with pytest.raises(ValueError, match="JSON must contain a dictionary at root level"):
                loader.load_from_json(json_file)

    def test_load_from_json_invalid_nested_structure(self):
        """Test loading from JSON with invalid nested structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "invalid_nested.json"
            
            # Create JSON with invalid nested structure
            data = {
                "laptop": {"doc1": 3, "doc2": 1},
                "smartphone": "not a dictionary",  # Invalid
                "tablet": {"doc3": 2}
            }
            
            with open(json_file, 'w') as f:
                json.dump(data, f)
            
            loader = JudgmentLoader()
            judgments = loader.load_from_json(json_file)
            
            # Should skip invalid entries
            assert len(judgments) == 2
            assert "laptop" in judgments
            assert "tablet" in judgments
            assert "smartphone" not in judgments

    def test_load_judgments_with_queries(self):
        """Test loading queries and judgments from separate files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queries_file = Path(temp_dir) / "queries.txt"
            qrels_file = Path(temp_dir) / "qrels.txt"
            
            # Create queries file
            with open(queries_file, 'w') as f:
                f.write("1:laptop computers\n")
                f.write("2:smartphone reviews\n")
                f.write("3:tablet comparison\n")
            
            # Create qrels file
            with open(qrels_file, 'w') as f:
                f.write("1 0 doc1 3\n")
                f.write("1 0 doc2 1\n")
                f.write("2 0 doc3 2\n")
                f.write("3 0 doc4 1\n")
            
            loader = JudgmentLoader()
            queries, judgments = loader.load_judgments_with_queries(queries_file, qrels_file)
            
            assert len(queries) == 3
            assert "laptop computers" in queries
            assert "smartphone reviews" in queries
            assert "tablet comparison" in queries
            
            assert len(judgments) == 3
            assert judgments["laptop computers"]["doc1"] == 3
            assert judgments["smartphone reviews"]["doc3"] == 2
            assert judgments["tablet comparison"]["doc4"] == 1

    def test_load_judgments_with_queries_simple_format(self):
        """Test loading queries without explicit IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queries_file = Path(temp_dir) / "simple_queries.txt"
            qrels_file = Path(temp_dir) / "qrels.txt"
            
            # Create queries file without IDs
            with open(queries_file, 'w') as f:
                f.write("laptop computers\n")
                f.write("smartphone reviews\n")
            
            # Create qrels file using line numbers
            with open(qrels_file, 'w') as f:
                f.write("1 0 doc1 3\n")
                f.write("2 0 doc2 2\n")
            
            loader = JudgmentLoader()
            queries, judgments = loader.load_judgments_with_queries(queries_file, qrels_file)
            
            assert len(queries) == 2
            assert "laptop computers" in queries
            assert "smartphone reviews" in queries
            assert judgments["laptop computers"]["doc1"] == 3
            assert judgments["smartphone reviews"]["doc2"] == 2


class TestJudgmentSaver:
    """Test cases for the JudgmentSaver class."""

    def test_save_to_csv(self):
        """Test saving judgments to CSV format."""
        judgments = {
            "laptop": {"doc1": 3, "doc2": 1},
            "smartphone": {"doc3": 2, "doc4": 3}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "saved_judgments.csv"
            
            saver = JudgmentSaver()
            saver.save_to_csv(judgments, output_file)
            
            assert output_file.exists()
            
            # Verify content
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 4
            assert reader.fieldnames == ['query', 'document_id', 'relevance_score']
            
            # Check specific entries (order may vary)
            laptop_rows = [r for r in rows if r['query'] == 'laptop']
            smartphone_rows = [r for r in rows if r['query'] == 'smartphone']
            
            assert len(laptop_rows) == 2
            assert len(smartphone_rows) == 2

    def test_save_to_csv_with_metadata(self):
        """Test saving judgments to CSV with metadata column."""
        judgments = {
            "laptop": {"doc1": 3, "doc2": 1}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "judgments_with_metadata.csv"
            
            saver = JudgmentSaver()
            saver.save_to_csv(judgments, output_file, include_metadata=True)
            
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert 'metadata' in reader.fieldnames
            assert len(rows) == 2

    def test_save_to_trec_qrels(self):
        """Test saving judgments to TREC qrels format."""
        judgments = {
            "laptop": {"doc1": 3, "doc2": 1},
            "smartphone": {"doc3": 2}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "saved_qrels.txt"
            
            saver = JudgmentSaver()
            saver.save_to_trec_qrels(judgments, output_file)
            
            assert output_file.exists()
            
            # Verify content
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 3
            
            # Check format (query_id 0 doc_id relevance)
            for line in lines:
                parts = line.strip().split()
                assert len(parts) == 4
                assert parts[1] == '0'  # iteration column

    def test_save_to_trec_qrels_with_query_id_map(self):
        """Test saving to TREC qrels with custom query ID mapping."""
        judgments = {
            "laptop computers": {"doc1": 3},
            "smartphone reviews": {"doc2": 2}
        }
        
        query_id_map = {
            "laptop computers": "Q001",
            "smartphone reviews": "Q002"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "custom_qrels.txt"
            
            saver = JudgmentSaver()
            saver.save_to_trec_qrels(judgments, output_file, query_id_map)
            
            with open(output_file, 'r') as f:
                content = f.read()
            
            assert "Q001 0 doc1 3" in content
            assert "Q002 0 doc2 2" in content

    def test_save_to_json(self):
        """Test saving judgments to JSON format."""
        judgments = {
            "laptop": {"doc1": 3, "doc2": 1},
            "smartphone": {"doc3": 2, "doc4": 3}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "saved_judgments.json"
            
            saver = JudgmentSaver()
            saver.save_to_json(judgments, output_file)
            
            assert output_file.exists()
            
            # Verify content
            with open(output_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == judgments

    def test_save_to_json_custom_indent(self):
        """Test saving to JSON with custom indentation."""
        judgments = {"laptop": {"doc1": 3}}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "indented_judgments.json"
            
            saver = JudgmentSaver()
            saver.save_to_json(judgments, output_file, indent=4)
            
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Check that indentation is used
            assert "    " in content


class TestJudgmentValidator:
    """Test cases for the JudgmentValidator class."""

    def test_validate_judgments_valid(self):
        """Test validating valid judgments."""
        judgments = {
            "laptop": {"doc1": 3, "doc2": 1, "doc3": 0},
            "smartphone": {"doc4": 2, "doc5": 3}
        }
        
        validator = JudgmentValidator()
        report = validator.validate_judgments(judgments)
        
        assert report['valid'] is True
        assert len(report['errors']) == 0
        assert report['statistics']['total_queries'] == 2
        assert report['statistics']['total_judgments'] == 5

    def test_validate_judgments_empty(self):
        """Test validating empty judgments."""
        validator = JudgmentValidator()
        report = validator.validate_judgments({})
        
        assert report['valid'] is False
        assert "No judgments provided" in report['errors']

    def test_validate_judgments_non_integer_scores(self):
        """Test validating judgments with non-integer scores."""
        judgments = {
            "laptop": {"doc1": 3.5, "doc2": "invalid"},  # Non-integers
            "smartphone": {"doc3": 2}
        }
        
        validator = JudgmentValidator()
        report = validator.validate_judgments(judgments)
        
        assert report['valid'] is False
        assert len(report['errors']) == 2
        assert "Non-integer relevance score" in report['errors'][0]

    def test_validate_judgments_out_of_range(self):
        """Test validating judgments with out-of-range scores."""
        judgments = {
            "laptop": {"doc1": -1, "doc2": 5},  # Below min, above max
            "smartphone": {"doc3": 2}
        }
        
        validator = JudgmentValidator()
        report = validator.validate_judgments(
            judgments,
            min_relevance=0,
            max_relevance=3
        )
        
        assert report['valid'] is False
        assert len(report['errors']) == 2
        assert "below minimum" in report['errors'][0]
        assert "above maximum" in report['errors'][1]

    def test_validate_judgments_empty_query_judgments(self):
        """Test validating judgments with empty query judgments."""
        judgments = {
            "laptop": {"doc1": 3, "doc2": 1},
            "smartphone": {},  # Empty judgments
            "tablet": {"doc3": 2}
        }
        
        validator = JudgmentValidator()
        report = validator.validate_judgments(judgments)
        
        assert len(report['warnings']) > 0
        assert "smartphone" in report['statistics']['empty_queries']

    def test_validate_judgments_against_expected_queries(self):
        """Test validating judgments against expected queries list."""
        judgments = {
            "laptop": {"doc1": 3},
            "smartphone": {"doc2": 2}
            # Missing "tablet"
        }
        
        expected_queries = ["laptop", "smartphone", "tablet"]
        
        validator = JudgmentValidator()
        report = validator.validate_judgments(judgments, queries=expected_queries)
        
        assert len(report['warnings']) > 0
        assert "tablet" in report['statistics']['queries_without_judgments']

    def test_validate_judgments_extra_queries(self):
        """Test validating judgments with extra queries."""
        judgments = {
            "laptop": {"doc1": 3},
            "smartphone": {"doc2": 2},
            "tablet": {"doc3": 1}  # Extra query
        }
        
        expected_queries = ["laptop", "smartphone"]
        
        validator = JudgmentValidator()
        report = validator.validate_judgments(judgments, queries=expected_queries)
        
        assert len(report['warnings']) > 0
        assert "Extra queries" in report['warnings'][0]

    def test_validate_judgments_relevance_distribution(self):
        """Test that relevance distribution is calculated correctly."""
        judgments = {
            "laptop": {"doc1": 3, "doc2": 1, "doc3": 0},
            "smartphone": {"doc4": 3, "doc5": 1}
        }
        
        validator = JudgmentValidator()
        report = validator.validate_judgments(judgments)
        
        distribution = report['statistics']['relevance_distribution']
        assert distribution[0] == 1  # One score of 0
        assert distribution[1] == 2  # Two scores of 1
        assert distribution[3] == 2  # Two scores of 3

    def test_suggest_judgment_scale_binary(self):
        """Test suggesting scale for binary judgments."""
        judgments = {
            "laptop": {"doc1": 0, "doc2": 1},
            "smartphone": {"doc3": 1, "doc4": 0}
        }
        
        validator = JudgmentValidator()
        analysis = validator.suggest_judgment_scale(judgments)
        
        assert analysis["min_score"] == 0
        assert analysis["max_score"] == 1
        assert analysis["unique_scores"] == [0, 1]
        assert "Binary relevance" in analysis["suggested_scale"]

    def test_suggest_judgment_scale_three_point(self):
        """Test suggesting scale for 3-point judgments."""
        judgments = {
            "laptop": {"doc1": 0, "doc2": 1, "doc3": 2},
            "smartphone": {"doc4": 1, "doc5": 2}
        }
        
        validator = JudgmentValidator()
        analysis = validator.suggest_judgment_scale(judgments)
        
        assert analysis["unique_scores"] == [0, 1, 2]
        assert "3-point scale" in analysis["suggested_scale"]

    def test_suggest_judgment_scale_four_point(self):
        """Test suggesting scale for 4-point judgments."""
        judgments = {
            "laptop": {"doc1": 0, "doc2": 1, "doc3": 2, "doc4": 3}
        }
        
        validator = JudgmentValidator()
        analysis = validator.suggest_judgment_scale(judgments)
        
        assert analysis["unique_scores"] == [0, 1, 2, 3]
        assert "4-point scale" in analysis["suggested_scale"]

    def test_suggest_judgment_scale_continuous(self):
        """Test suggesting scale for continuous judgments."""
        judgments = {
            "laptop": {f"doc{i}": i for i in range(10)}  # 0-9 scale
        }
        
        validator = JudgmentValidator()
        analysis = validator.suggest_judgment_scale(judgments)
        
        assert len(analysis["unique_scores"]) == 10
        assert "Continuous scale" in analysis["suggested_scale"]

    def test_suggest_judgment_scale_empty(self):
        """Test suggesting scale for empty judgments."""
        validator = JudgmentValidator()
        analysis = validator.suggest_judgment_scale({})
        
        assert "No judgments to analyze" in analysis["suggestion"]


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_load_judgments_auto_csv(self):
        """Test auto-detecting CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "judgments.csv"
            
            csv_data = [
                ["query", "document_id", "relevance_score"],
                ["laptop", "doc1", "3"],
                ["smartphone", "doc2", "2"]
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            judgments = load_judgments(csv_file, format="auto")
            
            assert len(judgments) == 2
            assert judgments["laptop"]["doc1"] == 3

    def test_load_judgments_auto_json(self):
        """Test auto-detecting JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "judgments.json"
            
            data = {"laptop": {"doc1": 3}, "smartphone": {"doc2": 2}}
            
            with open(json_file, 'w') as f:
                json.dump(data, f)
            
            judgments = load_judgments(json_file, format="auto")
            
            assert len(judgments) == 2
            assert judgments["laptop"]["doc1"] == 3

    def test_load_judgments_explicit_format(self):
        """Test loading with explicitly specified format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_file = Path(temp_dir) / "judgments.txt"  # Non-standard extension
            
            csv_data = [
                ["query", "document_id", "relevance_score"],
                ["laptop", "doc1", "3"]
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            judgments = load_judgments(csv_file, format="csv")
            
            assert len(judgments) == 1
            assert judgments["laptop"]["doc1"] == 3

    def test_load_judgments_unsupported_format(self):
        """Test loading with unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "judgments.txt"
            file_path.touch()
            
            with pytest.raises(ValueError, match="Unsupported format"):
                load_judgments(file_path, format="unsupported")

    def test_save_judgments_auto_csv(self):
        """Test auto-detecting CSV format for saving."""
        judgments = {"laptop": {"doc1": 3}}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "judgments.csv"
            
            save_judgments(judgments, output_file, format="auto")
            
            assert output_file.exists()
            
            # Verify it's CSV format
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
