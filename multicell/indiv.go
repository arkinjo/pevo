package multicell

import (
	//	"errors"
	//	"fmt"
	"log"
	"math"
	//	"math/rand"
)

type Cell struct { //A 'cell' is characterized by its gene expression and phenotype
	Id         int
	E          Vec     // Env + noise
	F          Vec     // Epigenetic markers
	G          Vec     // Gene expression
	H          Vec     // Higher order complexes
	P          Vec     // Phenotype
	Pave, Pvar Vec     // moving average and variance of P
	Vave, Vvar float64 // moving average and variance of total Pvar
	PErr       float64 // ||e - p||_1
	NDevStep   int     // Developmental path length

}

type Body struct { //Do we want to reimplement this?
	Genome   Genome
	Cells    []Cell // Array of cells of different types
	PErr     float64
	NDevStep int
}

const (
	IAncEnv = iota // Previous env
	INovEnv        // Current env
	NBodies
)

const ( // Index for cell state vectors
	CellE = iota
	CellF
	CellG
	CellH
	CellP
)

type Indiv struct { //An individual as an unicellular organism
	Id         int
	DadId      int
	MomId      int
	Bodies     []Body  //IAncEnv, INovEnv (see above const.)
	Fit        float64 //Fitness with cues
	WagFit     float64 //Wagner relative fitness
	Plasticity float64 //Observed Plasticity
	Dp1e1      float64 // ||p(e1) - e1||
	Dp0e0      float64 // ||p(e0) - e0||
	Dp1e0      float64 // ||p(e1) - e0||
	Dp0e1      float64 // ||p(e0) - e1||
}

func NewCell(id int) Cell { //Creates a new cell given id of cell.
	e := NewVec(nenv)
	f := NewVec(ngenes)
	g := NewVec(ngenes)
	h := NewVec(ngenes)
	p := NewVec(nenv)
	pa := NewVec(nenv)
	pv := NewVec(nenv)
	cell := Cell{id, e, f, g, h, p, pa, pv, 0.0, 0.0, 0.0, 0}

	return cell
}

func (cell *Cell) Copy() Cell {
	cell1 := NewCell(cell.Id)
	copy(cell1.E, cell.E)
	copy(cell1.F, cell.F)
	copy(cell1.G, cell.G)
	copy(cell1.H, cell.H)
	copy(cell1.P, cell.P)
	copy(cell1.Pave, cell.Pave)
	copy(cell1.Pvar, cell.Pvar)
	cell1.Vave = cell.Vave
	cell1.Vvar = cell.Vvar
	cell1.PErr = cell.PErr
	cell1.NDevStep = cell.NDevStep

	return cell1
}

func (cell *Cell) GetState(ivec string, ibeg, iend int) Vec {
	switch ivec {
	case "E":
		return cell.E[ibeg:iend]
	case "F":
		return cell.F[ibeg:iend]
	case "G":
		return cell.G[ibeg:iend]
	case "H":
		return cell.H[ibeg:iend]
	case "P":
		return cell.P[ibeg:iend]
	default:
		log.Fatal("Cell.GetState: Unknown state vector")
	}

	return nil // never happens
}

func NewBody(ncells int) Body {
	genome := NewGenome()
	cells := make([]Cell, ncells)
	for id := range cells {
		cells[id] = NewCell(id) //Initialize each cell
	}
	return Body{genome, cells, 0, 0}
}

func (body *Body) Copy() Body {
	body1 := NewBody(ncells)
	body1.PErr = body.PErr
	body1.NDevStep = body.NDevStep
	body1.Genome = body.Genome.Copy()
	for i, cell := range body.Cells {
		body1.Cells[i] = cell.Copy()
	}
	return body1
}

func NewIndiv(id int) Indiv { //Creates a new individual
	bodies := make([]Body, NBodies)
	for i := range bodies {
		bodies[i] = NewBody(ncells)
	}

	indiv := Indiv{id, 0, 0, bodies, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

	return indiv
}

func (indiv *Indiv) Copy() Indiv { //Deep copier
	indiv1 := NewIndiv(indiv.Id)
	indiv1.DadId = indiv.DadId
	indiv1.MomId = indiv.MomId
	for i, body := range indiv.Bodies {
		indiv1.Bodies[i] = body.Copy()
	}
	indiv1.Fit = indiv.Fit
	indiv1.Plasticity = indiv.Plasticity
	indiv1.Dp1e1 = indiv.Dp1e1
	indiv1.Dp0e0 = indiv.Dp0e0
	indiv1.Dp1e0 = indiv.Dp1e0
	indiv1.Dp0e1 = indiv.Dp0e1

	return indiv1
}

func Mate(dad, mom *Indiv) (Indiv, Indiv) { //Generates offspring
	bodies0 := make([]Body, NBodies)
	for i := range bodies0 {
		bodies0[i] = NewBody(ncells)

	}

	bodies1 := make([]Body, NBodies)
	for i := range bodies1 {
		bodies1[i] = NewBody(ncells)
	}

	genome0 := dad.Bodies[INovEnv].Genome.Copy()
	genome1 := mom.Bodies[INovEnv].Genome.Copy()
	CrossoverSpmats(genome0.E, genome1.E)
	CrossoverSpmats(genome0.F, genome1.F)
	CrossoverSpmats(genome0.G, genome1.G)
	CrossoverSpmats(genome0.H, genome1.H)
	CrossoverSpmats(genome0.J, genome1.J)
	CrossoverSpmats(genome0.P, genome1.P)

	bodies0[IAncEnv].Genome = genome0
	bodies1[IAncEnv].Genome = genome1
	bodies0[INovEnv].Genome = genome0.Copy()
	bodies1[INovEnv].Genome = genome1.Copy()

	// Different mutations for Anc and Nov envs.
	bodies0[IAncEnv].Genome.Mutate()
	bodies1[IAncEnv].Genome.Mutate()
	bodies0[INovEnv].Genome.Mutate()
	bodies1[INovEnv].Genome.Mutate()

	kid0 := Indiv{dad.Id, dad.Id, mom.Id, bodies0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	kid1 := Indiv{mom.Id, dad.Id, mom.Id, bodies1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

	return kid0, kid1
}

func (indiv *Indiv) getPErr(ienv int) float64 {
	return indiv.Bodies[ienv].PErr
}

func (indiv *Indiv) getNDevStep(ienv int) int {
	return indiv.Bodies[ienv].NDevStep
}

func (indiv *Indiv) getFitness() float64 { //fitness in novel/present environment
	ndevstep := indiv.getNDevStep(INovEnv)
	// if ndevstep == maxDevStep {
	// 	return minFitness
	// }

	fdev := float64(ndevstep) / selDevStep
	ferr := indiv.getPErr(INovEnv) / selPErr
	rawfit := math.Exp(-(ferr + fdev))
	return rawfit
}

func getPlasticity(body0, body1 Body) float64 { //cue plasticity of individual
	d2 := 0.0
	for i, cell := range body0.Cells {
		d2 += Dist2Vecs(cell.P, body1.Cells[i].P)
	}

	return d2 / float64(ncells*nenv)
}

func getPEDiff(body Body, envs Cues) float64 {
	diff := 0.0
	for i, c := range body.Cells {
		diff += DistVecs1(c.P[0:nsel], envs[i][0:nsel])
	}
	return diff / float64(ncells*nsel)
}

func (cell *Cell) updatePEMA(pnew Vec) {
	tvar := 0.0
	for i, pi := range pnew {
		d := pi - cell.Pave[i]
		incr := alphaEMA * d
		cell.Pave[i] += incr
		cell.Pvar[i] = (1 - alphaEMA) * (cell.Pvar[i] + d*incr)
		tvar += cell.Pvar[i]
	}
	tvar /= float64(nenv)
	d := tvar - cell.Vave
	incr := alphaEMA * d
	cell.Vave += incr
	cell.Vvar = (1 - alphaEMA) * (cell.Vvar + d*incr)
}

func (cell *Cell) DevCell(G Genome, env Cue) Cell { //Develops a cell given cue
	cell.P = Zeroes(nenv) // just to make sure it's zeroes.
	cue := Zeroes(nenv)
	g0 := Zeroes(ngenes)
	f0 := Zeroes(ngenes)
	h0 := Zeroes(ngenes)
	p := Zeroes(nenv)

	e_p := NewVec(nenv) // = env - p0

	Ee := NewVec(ngenes)
	Gg := NewVec(ngenes)
	Hg := NewVec(ngenes)
	Jh := NewVec(ngenes)
	f1 := NewVec(ngenes)
	g1 := NewVec(ngenes)
	h1 := NewVec(ngenes)

	//  AddNoise2CueNormal(cell.E, env, devNoise)
	AddNoise2CueFlip(cell.E, env, devNoise)

	if with_cue {
		cue = cell.E
	}

	for nstep := 1; nstep <= maxDevStep; nstep++ {
		MultMatVec(Gg, G.G, g0)
		if withE { //Model with or without cues
			if pheno_feedback { //p-feedback is allowed
				DiffVecs(e_p, cue, p)
				MultMatVec(Ee, G.E, e_p)
			} else {
				MultMatVec(Ee, G.E, cue)
			}
			AddVecs(f1, Gg, Ee)
		} else {
			copy(f1, Gg)
		}
		//		fa, fv := AveSD(f1)
		//		fmt.Println("Indiv.DevCell f1=", aveabs(f1), fa, fv)
		if withF { //Allow or disallow epigenetic layer
			DrawPFnVec(sigmaf, f1)
			MultMatVec(g1, G.F, f1)
		} else { //Remove epigenetic layer if false
			copy(g1, f1)
		}
		//		ga, gv := AveSD(g1)
		//		fmt.Println("Indiv.DevCell g1=", aveabs(g1), ga, gv)
		DrawPFnVec(sigmag, g1)
		if withH {
			MultMatVec(Hg, G.H, g1)
			if withJ {
				MultMatVec(Jh, G.J, h0)
				AddVecs(h1, Hg, Jh)
			} else {
				copy(h1, g1)
			}
			//	ha, hv := AveSD(h1)
			//fmt.Println("Indiv.DevCell h1=", aveabs(h1), ha, hv)
			DrawPFnVec(sigmah, h1)
		} else {
			copy(h1, g1) //identity map
		}

		MultMatVec(p, G.P, h1)
		//pa, pv := AveSD(p)
		//fmt.Println("Indiv.DevCell p=", aveabs(p), pa, pv)
		DrawPFnVec(sigmap, p)

		copy(f0, f1)
		copy(g0, g1)
		copy(h0, h1)
		cell.updatePEMA(p)

		//		fmt.Println("DevCell_var: ", nstep, cell.Vvar)
		cell.NDevStep = nstep
		if cell.Vvar < epsDev {
			break
		}
	}
	copy(cell.F, f1)
	copy(cell.G, g1)
	copy(cell.H, h1)
	copy(cell.P, p)
	cell.PErr = DistVecs1(cell.Pave[0:nsel], env[0:nsel]) / (2.0 * float64(nsel))

	return *cell
}

func (body *Body) DevBody(envs Cues) Body {
	sse := 0.0
	maxdev := 0

	for i, cell := range body.Cells {
		body.Cells[i] = cell.DevCell(body.Genome, envs[i])
		sse += cell.PErr
		if cell.NDevStep > maxdev {
			maxdev = cell.NDevStep
		}
		if cell.NDevStep > maxDevStep {
			log.Println("NDevStep greater than limit: ", cell.NDevStep)
		}
	}

	body.PErr = sse / float64(ncells*nsel)
	body.NDevStep = maxdev

	return *body
}

func (indiv *Indiv) Develop(ancenvs, novenvs Cues) Indiv { //Compare developmental process under different conditions
	indiv.Bodies[IAncEnv].DevBody(ancenvs)
	indiv.Bodies[INovEnv].DevBody(novenvs)

	indiv.Fit = indiv.getFitness()

	indiv.Plasticity = getPlasticity(indiv.Bodies[IAncEnv], indiv.Bodies[INovEnv])
	indiv.Dp1e1 = getPEDiff(indiv.Bodies[INovEnv], novenvs)
	indiv.Dp0e0 = getPEDiff(indiv.Bodies[IAncEnv], ancenvs)
	indiv.Dp1e0 = getPEDiff(indiv.Bodies[INovEnv], ancenvs)
	indiv.Dp0e1 = getPEDiff(indiv.Bodies[IAncEnv], novenvs)
	return *indiv
}
