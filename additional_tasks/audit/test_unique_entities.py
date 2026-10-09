"""Document-level entity counts reuse production identities and exclusions."""
import unittest
from collections import Counter

from additional_tasks.audit.audit_registry import candidates, unique_entity_counts


class UniqueEntityTests(unittest.TestCase):
    def test_repeated_and_inflected_person_and_email(self):
        text = ("Позивач Петренко Іван Іванович. Заява Петренка Івана Івановича. "
                "Відповідач Коваль Олег Петрович. "
                "Email person@example.com, повтор person@example.com.")
        found = candidates(text, mode="regex")
        self.assertEqual(Counter(c["type"] for c in found if not c["excluded"]),
                         {"ОСОБА": 3, "ІНФОРМАЦІЯ": 2})
        self.assertEqual(unique_entity_counts(found), {"ОСОБА": 2, "ІНФОРМАЦІЯ": 1})

    def test_service_only_identity_is_not_counted(self):
        found = candidates("Суддя Петренко Іван Іванович. Позивач Коваль Олег Петрович.", mode="regex")
        self.assertTrue(any(c["excluded"] for c in found))
        self.assertEqual(unique_entity_counts(found), {"ОСОБА": 1})

    def test_visible_mention_survives_excluded_mention_in_same_group(self):
        found = [{"type": "ОСОБА", "pseudonym": "ОСОБА_1", "excluded": excluded}
                 for excluded in (True, False, False)]
        self.assertEqual(unique_entity_counts(found), {"ОСОБА": 1})

    def test_document_counts_are_summed_without_cross_document_linking(self):
        counts = Counter()
        for text in ("Позивач Петренко Іван Іванович.", "Відповідач Петренко Іван Іванович."):
            counts.update(unique_entity_counts(candidates(text, mode="regex")))
        self.assertEqual(counts, {"ОСОБА": 2})

    def test_empty_and_raw_candidates(self):
        self.assertEqual(unique_entity_counts([]), {})
        self.assertIsNone(unique_entity_counts([{"type": "ОСОБА", "excluded": False}]))


if __name__ == "__main__":
    unittest.main()
