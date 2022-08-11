package multicell

// Canonical Polyadic Decomposition of 3-tensors.
/*
   The ALS1-CPO Algorithm in
   Mikael Sorensen, Lieven de Lathauwer, Pierre Comon, Sylvie Icart, Luc Deneire.
   Canonical Polyadic decomposition with a Columnwise Orthonormal Factor Matrix.
   SIAM Journal on Matrix Analysis and Applications, 2012, 33 (4), pp.1190-1213.
   ( https:/doi.org/10.1137/110830034 )
*/

import (
	//	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	//	"gonum.org/v1/gonum/mat"
)

const epsconv = 1e-6

type ElemsCPD struct {
	SVal float64
	Axes []Vec
}

type CPDArray []ElemsCPD

// X_{(1)}^T
func tflatten0(ten Tensor3) Dmat {
	len0 := len(ten)
	len1 := len(ten[0])
	len2 := len(ten[0][0])

	mat := make([]Vec, len2) // (len0*len1) X len2
	for k := 0; k < len2; k++ {
		for i := 0; i < len0; i++ {
			for j := 0; j < len1; j++ {
				mat[k] = append(mat[k], ten[i][j][k])
			}
		}
	}
	return mat
}

func tflatten1(ten Tensor3) Dmat {
	len0 := len(ten)
	len1 := len(ten[0])
	len2 := len(ten[0][0])

	mat := make([]Vec, len0) // (len1*len2) X len0

	for i := 0; i < len0; i++ {
		for j := 0; j < len1; j++ {
			for k := 0; k < len2; k++ {
				mat[i] = append(mat[i], ten[i][j][k])
			}
		}
	}
	return mat
}

func tflatten2(ten Tensor3) Dmat {
	len0 := len(ten)
	len1 := len(ten[0])
	len2 := len(ten[0][0])

	mat := make([]Vec, len1) // (len2*len0) X len1

	for j := 0; j < len1; j++ {
		for k := 0; k < len2; k++ {
			for i := 0; i < len0; i++ {
				mat[j] = append(mat[j], ten[i][j][k])
			}
		}
	}
	return mat
}

// The Khatri-Rao Product between KxN and KxN matrices
func KR_Product(a, b Dmat) Dmat {
	lena := len(a)
	lenb := len(b)
	lenc := len(a[0]) // should be equal to len(b[0])

	mat := NewDmat(lena*lenb, lenc)

	for i := 0; i < lena; i++ {
		ai := a[i]
		for j := 0; j < lenb; j++ {
			ind := i*lenb + j
			bj := b[j]
			for k := 0; k < lenc; k++ {
				mat[ind][k] = ai[k] * bj[k]
			}
		}
	}

	return mat
}

func NormDiag2(a Dmat) Vec {
	len0 := len(a)
	rank := len(a[0])
	dvec := NewVec(rank)

	for j := 0; j < rank; j++ {
		d := 0.0
		for i := 0; i < len0; i++ {
			v := a[i][j]
			d += v * v
		}
		if d > 0.0 {
			dvec[j] = 1.0 / d
		} else {
			dvec[j] = 0.0
		}
	}

	return dvec
}

func RandomInit(a Dmat, maxcol int) {
	ncol := len(a[0])
	if ncol > maxcol {
		ncol = maxcol
	}

	sd := 1.0 / math.Sqrt(float64(len(a)))
	for i := range a {
		for j := 0; j < ncol; j++ {
			a[i][j] = sd * rand.NormFloat64()
		}
	}
}

func checkDiffDmats(m0, m1 Dmat) float64 {
	dev := 0.0
	mag := 0.0
	for i, mi := range m1 {
		for j, mij := range mi {
			d := mij - m0[i][j]
			dev += d * d
			mag += mij * mij
		}
	}
	return dev / mag
}

func CheckCPD(ten Tensor3, cpd []ElemsCPD) (float64, float64) {
	dev := 0.0
	mag0 := 0.0
	mag1 := 0.0
	for i, ti := range ten {
		for j, tij := range ti {
			for k, t := range tij {
				mag0 += t * t
				v := 0.0
				for _, c := range cpd {
					v += c.SVal * c.Axes[0][i] * c.Axes[1][j] * c.Axes[2][k]
				}
				d := t - v
				dev += d * d
				mag1 += v * v
			}
		}
	}

	return dev / mag0, mag1 / mag0
}

// Canonical Polyadic Decomposition
// X = S*(A1 x A2 x A3) where A3 is orthogonal.
func GetCPDO(ten Tensor3, dir1, dir2 Vec, maxiter, rank int) []ElemsCPD {
	len0 := len(ten)
	len1 := len(ten[0])
	len2 := len(ten[0][0])

	if len0 < rank {
		rank = len0
	}
	if len1 < rank {
		rank = len1
	}
	if len2 < rank {
		rank = len2
	}

	a0 := NewDmat(len0, rank)
	a1 := NewDmat(len1, rank)
	a2 := NewDmat(len2, rank)
	ta2 := NewDmat(len2, rank)

	// Initialize
	RandomInit(a0, rank)
	RandomInit(a1, rank)
	RandomInit(a2, rank)

	xt0 := tflatten0(ten) // len2 x (len0*len1)
	xt1 := tflatten1(ten) // len0 x (len1*len2)
	tx2 := tflatten2(ten) // len1 x (len2*len0)

	for istep := 1; istep <= maxiter; istep++ {
		pa01 := KR_Product(a0, a1)   // (len0*len1) x rank
		tmat := MultDmats(xt0, pa01) // len2 x rank

		// Step 1: Update a0.
		U, _, V := GetSVD(tmat) // U: len2 x rank; V: rank x rank

		// Step 2: update a2.
		ResetDmat(ta2)
		for i := 0; i < len2; i++ {
			for j := 0; j < rank; j++ {
				for k := 0; k < rank; k++ {
					ta2[i][k] += U.At(i, j) * V.At(k, j)
				}
			}
		}
		// check convergence
		dev2 := checkDiffDmats(ta2, a2)

		CopyDmat(a2, ta2)

		// Step 3: Update a0.
		pa12 := KR_Product(a1, a2)  // (len1*len2) x rank
		ta0 := MultDmats(xt1, pa12) // len0 x rank
		d1 := NormDiag2(a1)         // rank
		for i := 0; i < len0; i++ {
			for j := 0; j < rank; j++ {
				a0[i][j] = ta0[i][j] * d1[j]
			}
		}

		// Step 4: Normalize a0.
		d0 := NormDiag2(a0)
		for i := 0; i < len0; i++ {
			for j := 0; j < rank; j++ {
				a0[i][j] *= math.Sqrt(d0[j])
			}
		}

		// Step 5: Update a1.
		pa20 := KR_Product(a2, a0)  // (len2*len0) x rank
		ta1 := MultDmats(tx2, pa20) // len1 x rank
		CopyDmat(a1, ta1)

		log.Println("GetCPDO:", istep, dev2)
		if dev2 < epsconv {
			log.Println("GetCPDO: Converged!")
			break
		}
	}

	sigma := NewVec(rank)
	for r := 0; r < rank; r++ {
		s := 0.0
		for i := 0; i < len1; i++ {
			s += a1[i][r] * a1[i][r]
		}
		sigma[r] = math.Sqrt(s)
		for i := 0; i < len1; i++ {
			a1[i][r] /= sigma[r]
		}

	}

	lst := make([]ElemsCPD, rank)
	for r := 0; r < rank; r++ {
		lst[r].SVal = sigma[r]
		lst[r].Axes = make([]Vec, 3)

		for i := 0; i < len0; i++ {
			lst[r].Axes[0] = append(lst[r].Axes[0], a0[i][r])
		}
		for i := 0; i < len1; i++ {
			lst[r].Axes[1] = append(lst[r].Axes[1], a1[i][r])
		}
		for i := 0; i < len2; i++ {
			lst[r].Axes[2] = append(lst[r].Axes[2], a2[i][r])
		}
	}

	sort.Slice(lst, func(i, j int) bool {
		return lst[i].SVal > lst[j].SVal
	})

	// Fix the sign of singular vectors.
	for r := 0; r < rank; r++ {
		d1 := DotVecs(dir1, lst[r].Axes[1])
		d2 := DotVecs(dir2, lst[r].Axes[2])
		if d1 < 0.0 {
			for i := range lst[r].Axes[1] {
				lst[r].Axes[1][i] *= -1.0
			}
			if d2 < 0.0 {
				for i := range lst[r].Axes[2] {
					lst[r].Axes[2][i] *= -1.0
				}
			} else {
				for i := range lst[r].Axes[0] {
					lst[r].Axes[0][i] *= -1.0
				}
			}
		} else if d2 < 0.0 {
			for i := range lst[r].Axes[2] {
				lst[r].Axes[2][i] *= -1.0
			}
			for i := range lst[r].Axes[0] {
				lst[r].Axes[0][i] *= -1.0
			}
		}

	}

	return lst
}
