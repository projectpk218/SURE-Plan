import unittest
from rapid_order_overview import completion_parts


class OrderOverviewTests(unittest.TestCase):
    def order(self, total=2000, before=800, today=100):
        return {"Original Quantity": total, "Completed Before Today": before, "Actual Production Today": today}

    def test_parts_reconcile_with_whole_order(self):
        total, before, today, left = completion_parts(self.order())
        self.assertEqual((total, before, today, left), (2000, 800, 100, 1100))
        self.assertEqual(before + today + left, total)

    def test_zero_and_completed_orders(self):
        self.assertEqual(completion_parts(self.order(before=0, today=0))[-1], 2000)
        self.assertEqual(completion_parts(self.order(before=1900, today=100))[-1], 0)

    def test_invalid_quantities_do_not_generate_a_misleading_ring(self):
        for order in (self.order(total=0), self.order(today=-1), self.order(before=2000)):
            with self.assertRaises(ValueError):
                completion_parts(order)
