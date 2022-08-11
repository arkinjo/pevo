package multicell

import (
	"log"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type elem_tensor3 struct {
	I, J, K int
	V       float64
}

type CoreTensor3 []elem_tensor3

func NewTensor3(n0, n1, n2 int) Tensor3 {
	ten := make([]Dmat, n0)
	for i := range ten {
		mat := make([]Vec, n1)
		for i := range mat {
			mat[i] = NewVec(n2)
		}
		ten[i] = mat
	}
	return ten
}

func flatten0(ten Tensor3) Dmat {
	len0 := len(ten)
	len1 := len(ten[0])
	len2 := len(ten[0][0])

	mat := make([]Vec, len0)

	for i := 0; i < len0; i++ {
		for j := 0; j < len1; j++ {
			for k := 0; k < len2; k++ {
				mat[i] = append(mat[i], ten[i][j][k])
			}
		}
	}

	return mat
}

func flatten1(ten Tensor3) Dmat {
	len0 := len(ten)
	len1 := len(ten[0])
	len2 := len(ten[0][0])

	mat := make([]Vec, len1)

	for j := 0; j < len1; j++ {
		for i := 0; i < len0; i++ {
			for k := 0; k < len2; k++ {
				mat[j] = append(mat[j], ten[i][j][k])
			}
		}
	}

	return mat
}

func flatten2(ten Tensor3) Dmat {
	len0 := len(ten)
	len1 := len(ten[0])
	len2 := len(ten[0][0])

	mat := make([]Vec, len2)

	for k := 0; k < len2; k++ {
		for i := 0; i < len0; i++ {
			for j := 0; j < len1; j++ {
				mat[k] = append(mat[k], ten[i][j][k])
			}
		}
	}

	return mat
}

func FlattenTensor3(ten Tensor3, ind int) Dmat {
	if ind == 0 {
		return flatten0(ten)
	} else if ind == 1 {
		return flatten1(ten)
	} else {
		return flatten2(ten)
	}
}

// Returns a compact core tensor and orthonormal basis.
func GetHOSVD(ten Tensor3) (Tensor3, *mat.Dense, *mat.Dense, *mat.Dense) {
	len0 := len(ten)
	len1 := len(ten[0])
	len2 := len(ten[0][0])

	mat0 := flatten0(ten)
	u0, s0, _ := GetSVD(mat0)
	mat1 := flatten1(ten)
	u1, s1, _ := GetSVD(mat1)
	mat2 := flatten2(ten)
	u2, s2, _ := GetSVD(mat2)

	rank0 := len(s0)
	rank1 := len(s1)
	rank2 := len(s2)

	score := NewTensor3(rank0, rank1, rank2)
	for i := 0; i < rank0; i++ {

		for j := 0; j < rank1; j++ {
			for k := 0; k < rank2; k++ {
				go func(i, j, k int) {
					ui := u0.ColView(i)
					uj := u1.ColView(j)
					uk := u2.ColView(k)
					for i1 := 0; i1 < len0; i1++ {
						for j1 := 0; j1 < len1; j1++ {
							for k1 := 0; k1 < len2; k1++ {
								score[i][j][k] += ten[i1][j1][k1] * ui.AtVec(i1) * uj.AtVec(j1) * uk.AtVec(k1)
							}
						}
					}
				}(i, j, k)
			}
		}
	}
	return score, u0, u1, u2
}

// Interlaced Computation; supposed to be significantly faster.
// Should be len0 <= len1 <= len2
func GetFastHOSVD(ten Tensor3) (CoreTensor3, *mat.Dense, *mat.Dense, *mat.Dense) {

	len0 := len(ten)
	len1 := len(ten[0])
	len2 := len(ten[0][0])

	mat0 := flatten0(ten)
	u0, s0, _ := GetSVD_opt(mat0, mat.SVDThinU)
	rank0 := len(s0)
	log.Println("GetFastHOSVD rank0=", rank0)

	A0 := NewTensor3(rank0, len1, len2)

	for i := 0; i < rank0; i++ {
		log.Println("step 1", i)
		ui := u0.ColView(i)
		for j := 0; j < len1; j++ {
			for k := 0; k < len2; k++ {
				go func(i, j, k int) {
					for l := 0; l < len0; l++ {
						A0[i][j][k] += ten[l][j][k] * ui.AtVec(l)
					}
				}(i, j, k)
			}
		}
	}

	mat1 := flatten1(A0)
	u1, s1, _ := GetSVD_opt(mat1, mat.SVDThinU)
	rank1 := len(s1)
	log.Println("GetFastHOSVD rank1=", rank1)

	if rank1 > rank0 {
		rank1 = rank0
		log.Println("GetFastHOSVD rank1 truncated to", rank1)
	}
	A1 := NewTensor3(rank0, rank1, len2)

	for i := 0; i < rank0; i++ {
		log.Println("step 2", i)
		for j := 0; j < rank1; j++ {
			uj := u1.ColView(j)
			for k := 0; k < len2; k++ {
				go func(i, j, k int) {
					for l := 0; l < len1; l++ {
						A1[i][j][k] += A0[i][l][k] * uj.AtVec(l)
					}
				}(i, j, k)
			}
		}
	}

	mat2 := flatten2(A1)
	u2, s2, _ := GetSVD_opt(mat2, mat.SVDThinU)
	rank2 := len(s2)
	log.Println("GetFastHOSVD rank2=", rank2)
	if rank2 > rank1 {
		rank2 = rank1
		log.Println("GetFastHOSVD rank2 truncated to", rank2)
	}

	A2 := NewTensor3(rank0, rank1, rank2)

	for i := 0; i < rank0; i++ {
		log.Println("step 3", i)
		for j := 0; j < rank1; j++ {
			for k := 0; k < rank2; k++ {
				uk := u2.ColView(k)
				go func(i, j, k int) {
					for l := 0; l < len2; l++ {
						A2[i][j][k] += A1[i][j][l] * uk.AtVec(l)
					}
				}(i, j, k)
			}
		}
	}

	vals := make([]elem_tensor3, 0)
	for i := 0; i < rank0; i++ {
		for j := 0; j < rank1; j++ {
			for k := 0; k < rank2; k++ {
				vals = append(vals, elem_tensor3{i, j, k, A2[i][j][k]})
			}
		}
	}
	sort.Slice(vals, func(i, j int) bool {
		return math.Abs(vals[i].V) > math.Abs(vals[j].V)
	})

	return vals, u0, u1, u2
}

func GetCrossCov3(vecs0, vecs1, vecs2 []Vec, submean0, submean1, submean2 bool) (Vec, Vec, Vec, Tensor3) {
	cv0 := GetMeanVec(vecs0)
	cv1 := GetMeanVec(vecs1)
	cv2 := GetMeanVec(vecs2)
	len0, len1, len2 := len(cv0), len(cv1), len(cv2)

	cov := NewTensor3(len0, len1, len2)

	var d0, d1, d2 float64

	for n := range vecs0 {
		for i, c0 := range cv0 {
			if submean0 {
				d0 = vecs0[n][i] - c0
			} else {
				d0 = vecs0[n][i]
			}
			for j, c1 := range cv1 {
				if submean1 {
					d1 = vecs1[n][j] - c1
				} else {
					d1 = vecs1[n][j]
				}
				for k, c2 := range cv2 {
					if submean2 {
						d2 = vecs2[n][k] - c2
					} else {
						d2 = vecs2[n][k]
					}
					cov[i][j][k] += d0 * d1 * d2
				}
			}
		}
	}

	fn := 1.0 / float64(len(vecs0))
	for i := range cv0 {
		for j := range cv1 {
			for k := range cv2 {
				cov[i][j][k] *= fn
			}
		}
	}

	return cv0, cv1, cv2, cov
}
