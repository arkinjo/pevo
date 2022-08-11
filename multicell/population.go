package multicell

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	//	"gonum.org/v1/gonum/mat"
)

type Population struct { //Population of individuals
	Params  Settings
	Gen     int
	NovEnvs Cues //Novel Environment
	AncEnvs Cues // Ancestral Environment
	Indivs  []Indiv
}

type PopStats struct { // Statistics of population (mean values of quantities of interest)
	PEDot      float64 // (delta P, delta E) dot product
	PErr1      float64 // <|| p(e1) - e1 ||> (Nov)
	PErr0      float64 // <|| p(e0) - e0 ||> (Anc)
	PED10      float64 // <|| p(e1) - e0 ||> (Nov-Anc)
	PED01      float64 // <|| p(e0) - e1 ||> (Anc-Nov)
	WagFit     float64
	Fitness    float64
	Plasticity float64
	Div        float64
	NDevStep   float64
}

func (pop *Population) GetStats() PopStats {
	var stats PopStats
	mf := 0.0
	maxfit := 0.0
	merr1 := 0.0
	merr0 := 0.0
	md10 := 0.0
	md01 := 0.0
	ndev := 0
	mop := 0.0 // mean observed plasticity
	fn := float64(len(pop.Indivs))
	pa := NewCues(ncells, nenv)
	pv := NewCues(ncells, nenv)

	denv := 0.0
	for i, env := range pop.NovEnvs { //To normalize wrt change in environment cue
		denv += DistVecs1(env, pop.AncEnvs[i])
	}

	for _, indiv := range pop.Indivs {
		merr1 += indiv.Dp1e1
		merr0 += indiv.Dp0e0
		md10 += indiv.Dp1e0
		md01 += indiv.Dp0e1
		mf += indiv.Fit
		mop += indiv.Plasticity
		ndev += indiv.Bodies[INovEnv].NDevStep

		if indiv.Fit > maxfit {
			maxfit = indiv.Fit
		}
		for i, cell := range indiv.Bodies[INovEnv].Cells {
			for j, t := range cell.P {
				pa[i][j] += t
			}
		}

	}
	for i, p := range pa {
		for j := range p {
			pa[i][j] /= fn

		}
	}
	for _, indiv := range pop.Indivs {
		for i, cell := range indiv.Bodies[INovEnv].Cells {
			for j, t := range cell.P {
				d := pa[i][j] - t
				pv[i][j] += d * d
			}
		}
	}

	div := 0.0
	for _, pi := range pv {
		for _, t := range pi {
			div += t / fn
		}
	}

	env0 := FlattenEnvs(GetSelEnvs(pop.AncEnvs))
	env1 := FlattenEnvs(GetSelEnvs(pop.NovEnvs))
	lenP := len(env1)
	dirE := NewVec(lenP)
	DiffVecs(dirE, env1, env0)
	NormalizeVec(dirE)

	mp1 := GetMeanVec(pop.GetFlatStateVec("P", 1, 0, nsel))
	dirP := NewVec(lenP)
	DiffVecs(dirP, mp1, env0)
	NormalizeVec(dirP)

	stats.PEDot = DotVecs(dirP, dirE)
	stats.PErr1 = merr1 / fn
	stats.PErr0 = merr0 / fn
	stats.PED10 = md10 / fn
	stats.PED01 = md01 / fn
	meanfit := mf / fn
	stats.Fitness = meanfit
	stats.WagFit = meanfit / maxfit
	stats.NDevStep = float64(ndev) / fn
	stats.Plasticity = mop / (fn * denv)
	stats.Div = div

	return stats
}

func NewPopulation(s Settings) Population {
	SetParams(s)

	envs0 := NewCues(s.NCells, s.NEnv)
	envs1 := NewCues(s.NCells, s.NEnv)

	indivs := make([]Indiv, s.MaxPop)
	for i := range indivs {
		indivs[i] = NewIndiv(i)
	}

	p := Population{Params: s, Gen: 0, AncEnvs: envs0, NovEnvs: envs1,
		Indivs: indivs}
	return p
}

func (pop *Population) FromJSON(filename string) {
	pop.ClearGenome()
	fin, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}

	byteValue, _ := ioutil.ReadAll(fin)
	err = json.Unmarshal(byteValue, pop)
	if err != nil {
		log.Fatal(err)
	}

	err = fin.Close()
	if err != nil {
		log.Fatal(err)
	}
	log.Println("Successfully imported population from", filename)
}

func (pop *Population) ToJSON(filename string) {
	jsonpop, err := json.Marshal(pop) //JSON encoding of population as byte array
	if err != nil {
		log.Fatal(err)
	}
	fout, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644) //create json file
	if err != nil {
		log.Fatal(err)
	}
	_, err = fout.Write(jsonpop)
	if err != nil {
		log.Fatal(err)
	}
	err = fout.Close()
	if err != nil {
		log.Fatal(err)
	}

	log.Println("Successfully exported population to", filename)
}

func (pop *Population) SetWagnerFitness() { //compute normalized fitness value similar to Wagner (1996).
	var mf float64
	for _, indiv := range pop.Indivs {
		if mf < indiv.Fit {
			mf = indiv.Fit
		}
	}
	for i, indiv := range pop.Indivs {
		pop.Indivs[i].WagFit = math.Max(indiv.Fit/mf, minWagnerFitness)
		//Zero fitness individuals that don't converge can still reproduce
	}
}

func (pop *Population) RandomizeGenome() {
	for _, indiv := range pop.Indivs { //Sets genome of every individual to

		indiv.Bodies[0].Genome.Randomize()
		indiv.Bodies[1].Genome = indiv.Bodies[0].Genome.Copy()
		indiv.Bodies[1].Genome.Mutate()
	}
}

func (pop *Population) ClearGenome() {
	for _, indiv := range pop.Indivs { //Sets genome of every individual to zero
		for i := range indiv.Bodies {
			indiv.Bodies[i].Genome.Clear()
		}
	}
}

func (pop *Population) Copy() Population {
	pop1 := NewPopulation(pop.Params)
	pop1.Params = pop.Params
	pop1.Gen = pop.Gen
	pop1.NovEnvs = CopyCues(pop.NovEnvs)
	pop1.AncEnvs = CopyCues(pop.AncEnvs)
	for i, indiv := range pop.Indivs {
		pop1.Indivs[i] = indiv.Copy()
	}
	return pop1
}

func (pop *Population) GetMeanPhenotype(gen int) Cues { //elementwise average phenotype of population; output as slice instead of cue struct
	npop := len(pop.Indivs)
	MeanPhenotype := NewCues(ncells, nenv)
	pop.DevPop(gen)

	for _, indiv := range pop.Indivs {
		for i, c := range indiv.Bodies[INovEnv].Cells {
			for j, p := range c.P {
				MeanPhenotype[i][j] += p / float64(npop)
			}
		}
	}
	return MeanPhenotype
}

func (pop *Population) Selection(nNewPop int) []Indiv { //Selects parents for new population
	npop := len(pop.Indivs)
	parents := make([]Indiv, 0)
	ipop := 0
	cnt := 0
	for ipop < nNewPop && cnt < 1000*nNewPop {
		cnt += 1
		k := rand.Intn(npop)
		ind := pop.Indivs[k]
		r := rand.Float64()
		if r < ind.WagFit {
			parents = append(parents, ind)
			ipop += 1
		}
	}
	return parents
}

func (pop *Population) Reproduce(nNewPop int) Population { //Crossover
	parents := pop.Selection(nNewPop)
	nindivs := make([]Indiv, 0)
	npop := len(parents)

	for len(nindivs) < nNewPop { //Randomly reproduce among survivors
		k := rand.Intn(npop)
		l := rand.Intn(npop)
		dad := parents[k]
		mom := parents[l]
		kid0, kid1 := Mate(&dad, &mom)
		nindivs = append(nindivs, kid0)
		nindivs = append(nindivs, kid1)
		//ipop += 2
	}

	for i := range nindivs {
		nindivs[i].Id = i //Relabels individuals according to position in array
	}

	new_population := Population{pop.Params, 0, pop.NovEnvs, pop.AncEnvs, nindivs} //resets embryonic values to zero!

	return new_population

}

func (pop *Population) PairReproduce(nNewPop int) Population { //Crossover in ordered pairs; as in Wagner's
	var index int
	parents := pop.Selection(nNewPop)
	//nparents := len(parents)
	nindivs := make([]Indiv, 0)

	for index < nNewPop && len(nindivs) < nNewPop { //Forced reproduction in ordered pairs; may cause bugs when population has an odd number of survivors
		dad := parents[index]
		mom := parents[index+1]
		kid0, kid1 := Mate(&dad, &mom)
		nindivs = append(nindivs, kid0)
		nindivs = append(nindivs, kid1)
		index = len(nindivs) //update
	}

	for i := range nindivs {
		nindivs[i].Id = i //Relabels individuals according to position in array
	}

	new_population := Population{pop.Params, 0, pop.NovEnvs, pop.AncEnvs, nindivs} //resets embryonic values to zero!

	return new_population
}

func (pop *Population) SortPopIndivs() {
	sort.Slice(pop.Indivs, func(i, j int) bool { return pop.Indivs[i].Id < pop.Indivs[j].Id })
}

func (pop *Population) ChangeEnvs(denv int) {
	OldEnvs := CopyCues(pop.NovEnvs)
	pop.AncEnvs = OldEnvs
	pop.NovEnvs = ChangeEnvs(OldEnvs, denv)
}

func (pop *Population) SetRandomNovEnvs() {
	pop.NovEnvs = RandomEnvs(ncells, 0.5)
}

func (pop *Population) DevPop(gen int) Population {
	pop.Gen = gen

	ch := make(chan Indiv) //channels for parallelization
	for _, indiv := range pop.Indivs {
		go func(indiv Indiv) {
			ch <- indiv.Develop(pop.AncEnvs, pop.NovEnvs)
		}(indiv)
	}
	for i := range pop.Indivs {
		pop.Indivs[i] = <-ch //Update output results
	}

	pop.SetWagnerFitness()
	//We might need a sorter here.
	pop.SortPopIndivs()

	return *pop
}

//Records population trajectory and writes files
func (pop0 *Population) Evolve(test bool, ftraj *os.File, jsonout string, nstep, epoch int) Population {
	pop := *pop0

	fmt.Fprintln(ftraj, "#Epoch\tGen\tNpop\tPhenoEnvDot \tMeanErr1 \tMeanErr0 \tMeanDp1e0 \tMeanDp0e1 \tFitness \tWag_Fit \tObs_Plas \tDiversity \tNdev") //header

	for istep := 1; istep <= nstep; istep++ {
		pop.DevPop(istep)
		if test {
			if jsonout != "" { //Export JSON population of each generation in test mode
				filename := fmt.Sprintf("%s_%2.2d_%3.3d.json", jsonout, epoch, pop.Gen)
				pop.ToJSON(filename)
			}
		}

		pstat := pop.GetStats()
		popsize := len(pop.Indivs)

		fmt.Fprintf(ftraj, "%d\t%d\t%d\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n", epoch, istep, popsize, pstat.PEDot, pstat.PErr1, pstat.PErr0, pstat.PED10, pstat.PED01, pstat.Fitness, pstat.WagFit, pstat.Plasticity, pstat.Div, pstat.NDevStep)

		fmt.Printf("Evolve: %d\t<ME1>: %e\t<ME0>: %e\n", istep, pstat.PErr1, pstat.PErr0)

		pop = pop.PairReproduce(maxPop)
	}
	return pop
}

func (pop *Population) GetFlatStateVec(istate string, ienv, ibeg, iend int) Dmat {
	vs0 := make([]Vec, 0)
	for _, indiv := range pop.Indivs {
		tv0 := make([]float64, 0)
		for _, cell := range indiv.Bodies[ienv].Cells {
			tv0 = append(tv0, cell.GetState(istate, ibeg, iend)...)
		}
		vs0 = append(vs0, tv0)
	}

	return vs0
}

func (pop *Population) GetFlatGenome(IEnv int) Dmat {
	vs := make([]Vec, 0)
	for _, indiv := range pop.Indivs {
		tv := indiv.Bodies[IEnv].Genome.FlatVec()
		vs = append(vs, tv)
	}
	return vs
}
