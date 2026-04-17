import unittest

from wildfire_risk.build_labels import parse_mcd64a1_extent, parse_mcd64a1_filename


class MCD64A1HelperTest(unittest.TestCase):
    def test_parse_mcd64a1_filename(self) -> None:
        month, year, doy, tile_id = parse_mcd64a1_filename(
            "MCD64A1.A2020032.h10v04.061.2021309110419.hdf"
        )
        self.assertEqual(month, "2020-02")
        self.assertEqual(year, 2020)
        self.assertEqual(doy, 32)
        self.assertEqual(tile_id, "h10v04")

    def test_parse_mcd64a1_extent(self) -> None:
        struct_metadata = """
        GROUP=GridStructure
            GROUP=GRID_1
                UpperLeftPointMtrs=(-8895604.157328,5559752.598331)
                LowerRightMtrs=(-7783653.637661,4447802.078664)
            END_GROUP=GRID_1
        END_GROUP=GridStructure
        """
        ulx, uly, lrx, lry = parse_mcd64a1_extent(struct_metadata)
        self.assertAlmostEqual(ulx, -8895604.157328)
        self.assertAlmostEqual(uly, 5559752.598331)
        self.assertAlmostEqual(lrx, -7783653.637661)
        self.assertAlmostEqual(lry, 4447802.078664)


if __name__ == "__main__":
    unittest.main()
