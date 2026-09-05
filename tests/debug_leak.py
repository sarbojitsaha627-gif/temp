import asyncio
from sqlalchemy import select
from app.db.core import async_session
from app.db.models import Order, Refund, Dispute, Settlement, BankTransaction, ExceptionStaging

async def debug_leak():
    async with async_session() as session:
        orders = (await session.execute(select(Order))).scalars().all()
        refunds = (await session.execute(select(Refund))).scalars().all()
        disputes = (await session.execute(select(Dispute))).scalars().all()
        settlements = (await session.execute(select(Settlement))).scalars().all()
        bank_txns = (await session.execute(select(BankTransaction))).scalars().all()
        exceptions = (await session.execute(select(ExceptionStaging))).scalars().all()

        sum_order_gross = sum(o.gross_amount_paise for o in orders)
        sum_refunds = sum(r.amount_paise for r in refunds)
        sum_disputes = sum(d.amount_paise for d in disputes)
        
        sum_setl_net = sum(s.net_amount_paise for s in settlements)
        sum_setl_fee = sum(s.fee_base_paise for s in settlements)
        sum_setl_tax = sum(s.fee_tax_gst_paise for s in settlements)
        sum_bank_charges = sum(b.bank_charges_paise for b in bank_txns)
        sum_exception_residuals = sum(ex.variance_paise for ex in exceptions if ex.variance_paise)
        
        print(f"LHS components:")
        print(f"  orders: {sum_order_gross}")
        print(f"  refunds: -{sum_refunds}")
        print(f"  disputes: -{sum_disputes}")
        print(f"  LHS TOTAL: {sum_order_gross - sum_refunds - sum_disputes}")
        print()
        print(f"RHS components:")
        print(f"  setl_net: {sum_setl_net}")
        print(f"  setl_fee: {sum_setl_fee}")
        print(f"  setl_tax: {sum_setl_tax}")
        print(f"  bank_charges: {sum_bank_charges}")
        print(f"  exception_residuals: {sum_exception_residuals}")
        print(f"  RHS TOTAL: {sum_setl_net + sum_setl_fee + sum_setl_tax + sum_bank_charges + sum_exception_residuals}")
        print()
        
        for ex in exceptions:
            if ex.variance_paise:
                print(f"Exception {ex.category.name} ({ex.entity_type} {ex.entity_id}): {ex.variance_paise}")

if __name__ == "__main__":
    asyncio.run(debug_leak())
