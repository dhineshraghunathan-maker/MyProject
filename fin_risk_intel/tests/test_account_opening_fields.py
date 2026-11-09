import unittest

from src.data_gen import generate


class AccountOpeningFieldsTest(unittest.TestCase):
    def test_kyc_contains_required_identity_and_income_fields(self):
        data = generate()
        kyc = data["kyc"]
        required = [
            "declared_id_type",
            "declared_id_number",
            "declared_occupation",
            "declared_source_of_income",
            "declared_income_range",
            "declared_pep",
            "doc_id_type",
            "doc_id_number",
            "doc_occupation",
            "doc_source_of_income",
            "doc_income_range",
            "doc_pep",
        ]
        missing = [field for field in required if field not in kyc.columns]
        self.assertFalse(missing, f"Missing synthetic fields: {missing}")

        for field in required:
            self.assertTrue(kyc[field].notna().all(), f"Field {field} contains nulls")


if __name__ == "__main__":
    unittest.main()
