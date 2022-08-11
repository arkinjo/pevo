package multicell

import (
	//	"math"
	"math/rand"

	"gonum.org/v1/gonum/stat/distuv"
)

type Spmat struct {
	Ncol int                 // number of columns
	Mat  [](map[int]float64) // Sparse matrix is an array of maps.
}

func NewSpmat(nrow, ncol int) Spmat { //Initialize new sparse matrix
	mat := make([](map[int]float64), nrow)
	for i := range mat {
		mat[i] = make(map[int]float64)
	}
	return Spmat{ncol, mat}
}

func (sp *Spmat) Copy() Spmat {
	nsp := NewSpmat(len(sp.Mat), sp.Ncol)
	for i, m := range sp.Mat {
		for j, d := range m {
			nsp.Mat[i][j] = d
		}
	}
	return nsp
}

func (sp *Spmat) Randomize(density float64) { //Randomize entries of sparse matrix
	if density == 0 {
		return
	}

	density2 := density / 2
	for i := range sp.Mat {
		for j := 0; j < sp.Ncol; j++ {
			r := rand.Float64()
			if r < density2 {
				sp.Mat[i][j] = 1
			} else if r < density {
				sp.Mat[i][j] = -1
			}
		}
	}
}

func DiffSpmat(m1, m2 *Spmat) Spmat { //This function works fine
	d := NewSpmat(len(m1.Mat), m1.Ncol) //initialization
	ncol := m1.Ncol
	for i := range m1.Mat {
		for j := 0; j < ncol; j++ {
			d.Mat[i][j] = m1.Mat[i][j] - m2.Mat[i][j]
		}
	}

	return d
}

func (sp *Spmat) Scale(c float64) {
	for i, mi := range sp.Mat {
		for j := range mi {
			sp.Mat[i][j] *= c
		}
	}
}

func MultMatVec(vout Vec, mat Spmat, vin Vec) { //Matrix multiplication
	for i := range vout {
		vout[i] = 0.0
	}

	for i, m := range mat.Mat {
		for j, d := range m {
			vout[i] += d * vin[j]
		}
	}
	return
}

func MultMatVec_T(vout Vec, mat Spmat, vin Vec) { //Matrix transposition and then multiplication
	for i := range vout {
		vout[i] = 0.0
	}
	for i, m := range mat.Mat {
		vi := vin[i]
		for j, d := range m {
			vout[j] += d * vi
		}
	}

	return
}

func (mat *Spmat) mutateSpmat(density, mutrate float64) { //mutating a sparse matrix
	if density == 0.0 {
		return
	}

	nrow := len(mat.Mat)
	lambda := mutrate * float64(nrow*mat.Ncol)
	dist := distuv.Poisson{Lambda: lambda}
	nmut := int(dist.Rand())
	density2 := density * 0.5
	for n := 0; n < nmut; n++ {
		i := rand.Intn(nrow)
		j := rand.Intn(mat.Ncol)
		r := rand.Float64()
		delete(mat.Mat[i], j)
		if r < density2 {
			mat.Mat[i][j] = 1.0
		} else if r < density {
			mat.Mat[i][j] = -1.0
		}
	}

	//Note: This implementation has non-zero probability of choosing same element to be mutated twice.
	return
}

// point mutation
func (mat *Spmat) pMutateSpmat(density float64, irow, icol int) {
	if density == 0.0 {
		return
	}

	r := rand.Float64()
	delete(mat.Mat[irow], icol)
	if r < density/2 {
		mat.Mat[irow][icol] = 1.0
	} else if r < density {
		mat.Mat[irow][icol] = -1.0
	}
	return
}

func DotSpmats(mat0, mat1 Spmat) float64 {
	dot := 0.0
	for i, m := range mat0.Mat {
		for j, d := range m {
			dot += d * mat1.Mat[i][j]
		}
	}

	return dot
}

func CrossoverSpmats(mat0, mat1 Spmat) {
	for i, ri := range mat0.Mat {
		r := rand.Float64()
		if r < 0.5 {
			mat0.Mat[i] = mat1.Mat[i]
			mat1.Mat[i] = ri
		}
	}

}
