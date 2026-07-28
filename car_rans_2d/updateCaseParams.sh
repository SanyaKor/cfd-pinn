#!/bin/bash

# ---- INPUT ----

U=2
I=0.1
c_mu=0.09
scale=0.001

Lx_car=41.9
Ly_car=73
Lz_car=20

nu=0.000015

file_ic="0/include/initialConditions"
file_fc="system/forceCoeffs"

# ---- CALC ----

k=$(awk "BEGIN {print 1.5 * ($I * $U)^2}")
l=$(awk "BEGIN {print 0.07 * $Ly_car * $scale}")
omega=$(awk "BEGIN {print sqrt($k) / (($c_mu)^(0.25) * $l)}")

ARef=$(awk "BEGIN {print $Lx_car * $Lz_car * ($scale^2)}")
LRef=$(awk "BEGIN {print $Ly_car * $scale}")
Re=$(awk "BEGIN {print $Ly_car * $scale * $U / $nu}")

# ---- FORMAT ----

fmt() {
    printf "%.4f\n" "$1"
}

k_fmt=$(fmt "$k")
omega_fmt=$(fmt "$omega")
ARef_fmt=$(fmt "$ARef")
LRef_fmt=$(fmt "$LRef")
Re_fmt=$(fmt "$Re")

# ---- SET FUNCTIONS ----

set_scalar() {
    local file="$1"
    local key="$2"
    local value="$3"

    awk -v key="$key" -v value="$value" '
    $1 == key {
        sub(/[0-9.eE+-]+;/, value ";")
    }
    { print }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

set_vector() {
    local file="$1"
    local key="$2"
    local value="$3"

    awk -v key="$key" -v value="$value" '
    $1 == key {
        sub(/\([^)]+\)/, "(" value ")")
    }
    { print }
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

# ---- APPLY TO FILES ----

set_vector "$file_ic" flowVelocity "0 $U 0"
set_scalar "$file_ic" turbulentKE "$k_fmt"
set_scalar "$file_ic" turbulentOmega "$omega_fmt"

set_scalar "$file_fc" Aref "$ARef_fmt"
set_scalar "$file_fc" lRef "$LRef_fmt"
set_scalar "$file_fc" magUInf "$U"

# ---- OUTPUT ----

echo "===== PARAMETERS UPDATED ====="
echo "Re = $Re_fmt"
echo "U = $U"
echo "k = $k_fmt"
echo "omega = $omega_fmt"
echo "ARef = $ARef_fmt"
echo "LRef = $LRef_fmt"
echo "magUInf = $U"
