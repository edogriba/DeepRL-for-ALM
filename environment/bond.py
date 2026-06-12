import numpy as np

# Bond class
class Bond:
    def __init__(self, bond_type, nominal_value, maturity_months, coupon_dates):
        """
        Args:
            bond_type: Identifier for the bond type k
            nominal_value: Face value V^(k)
            maturity_months: Original maturity M^(k)
            coupon_dates: List of months when coupons are paid P_tau
        """
        self.bond_type = bond_type
        self.V = nominal_value
        self.M = maturity_months
        self.coupon_dates = coupon_dates

    def get_inflow_at_age(self, age, coupon_rate=0.0):
        """
        Calculates the cash flow at a specific month of the bond's life.
        age: month number (1 to M)
        coupon_rate: the specific annual yield locked at issuance.
        """
        
        # Calculate coupon payment per coupon date (assuming fixed coupons and regular intervals)
        months_per_payment = (self.coupon_dates[1] - self.coupon_dates[0]) if len(self.coupon_dates) > 1 else self.coupon_dates[0]
        monthly_coupon = (coupon_rate * self.V) * (months_per_payment / 12.0)
        
        inflow = 0.0
        
        # If the current age is a coupon date, add the coupon
        if age in self.coupon_dates:
            inflow += monthly_coupon
            
        # If the bond has reached maturity, add the face value (nominal)
        if age == self.M:
            inflow += self.V
        return inflow
    
            
    def print_info(self):
        print(f"Bond Type: {self.bond_type}")
        print(f" Nominal Value: {self.V}")
        print(f" Maturity: {self.M} months")
        print(f" Coupon Dates: {self.coupon_dates}")