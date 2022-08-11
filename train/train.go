package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/arkinjo/pevo/multicell"
)

var T_Filename string = "traj.dat"
var json_in string //JSON encoding of initial population; default to empty string
var json_out string = "popout"
var jfilename string

func main() {
	t0 := time.Now()
	testP := flag.Bool("test", false, "Test run or not")
	maxpopP := flag.Int("maxpop", 1000, "maximum number of individuals in population")
	ngenesP := flag.Int("ngenes", 200, "Number of genes")
	nenvP := flag.Int("nenv", 200, "Number of environmental cues/traits")
	nselP := flag.Int("nsel", 40, "Number of environmental cues/traits for selection")
	ncellsP := flag.Int("ncells", 1, "Number of cell types")
	withcueP := flag.Bool("cue", true, "With environmental cue")
	flayerP := flag.Bool("layerF", true, "Epigenetic layer")
	hlayerP := flag.Bool("layerH", true, "Higher order complexes")
	jlayerP := flag.Bool("layerJ", true, "Interactions in higher order interactions")
	pfbackP := flag.Bool("pfback", true, "Phenotype feedback to input")

	noiseP := flag.Float64("noise", 0.05, "Strength of environmental noise")
	mutP := flag.Float64("mut", 0.005, "Mutation rate")
	denEP := flag.Float64("dE", 0.02, "Density of E")
	denFP := flag.Float64("dF", 0.02, "Density of F")
	denGP := flag.Float64("dG", 0.02, "Density of G")
	denHP := flag.Float64("dH", 0.02, "Density of H")
	denJP := flag.Float64("dJ", 0.02, "Density of J")
	denPP := flag.Float64("dP", 0.02, "Density of P")

	seedPtr := flag.Int("seed", 13, "random seed")
	seed_cuePtr := flag.Int("seed_cue", 7, "random seed for environmental cue")
	epochPtr := flag.Int("nepoch", 20, "number of epochs")
	genPtr := flag.Int("ngen", 200, "number of generation/epoch")

	denvPtr := flag.Int("denv", 100, "magnitude of environmental change")
	tfilenamePtr := flag.String("traj_file", "traj.dat", "filename of trajectories")
	jsoninPtr := flag.String("jsonin", "", "json file of input population") //default to empty string
	jsonoutPtr := flag.String("jsonout", "popout", "json file of output population")
	flag.Parse()

	var settings = multicell.CurrentSettings()
	settings.MaxPop = *maxpopP
	settings.NGenes = *ngenesP
	settings.NEnv = *nenvP
	settings.NSel = *nselP
	settings.NCells = *ncellsP
	settings.WithCue = *withcueP
	settings.WithCue = *withcueP
	settings.FLayer = *flayerP
	settings.HLayer = *hlayerP
	settings.JLayer = *jlayerP
	settings.Pfback = *pfbackP
	settings.SDNoise = *noiseP
	settings.MutRate = *mutP
	settings.DensityE = *denEP
	settings.DensityF = *denFP
	settings.DensityG = *denGP
	settings.DensityH = *denHP
	settings.DensityJ = *denJP
	settings.DensityP = *denPP

	log.Println("seed=", *seedPtr, "seed_cue=", *seed_cuePtr)
	multicell.SetSeed(int64(*seedPtr))
	multicell.SetSeedCue(int64(*seed_cuePtr))

	maxepochs := *epochPtr
	epochlength := *genPtr
	denv := *denvPtr
	T_Filename = *tfilenamePtr
	json_in = *jsoninPtr
	json_out = *jsonoutPtr
	test_flag := *testP

	pop0 := multicell.NewPopulation(settings)

	if json_in != "" { //read input population as a json file, if given
		pop0.FromJSON(json_in)
	}

	pop0.Params.SDNoise = settings.SDNoise
	pop0.Params.MutRate = settings.MutRate
	multicell.SetParams(pop0.Params)
	if json_in == "" {
		fmt.Println("Randomizing initial population")
		pop0.RandomizeGenome()
	}

	ftraj, err := os.OpenFile(T_Filename, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644) //create file for recording trajectory
	if err != nil {
		log.Fatal(err)
	}

	popstart := pop0
	if json_in != "" {
		popstart.ChangeEnvs(denv)
	} else {
		popstart.SetRandomNovEnvs()
	}

	fmt.Println("Initialization of population complete")
	dtint := time.Since(t0)
	fmt.Println("Time taken for initialization : ", dtint)

	envtraj := make([]multicell.Cues, 1) //Trajectory of environment cue
	envtraj[0] = popstart.AncEnvs
	novvec := make([]bool, 0)

	log.Println("AncEnvs", 0, ":", popstart.AncEnvs)
	for epoch := 1; epoch <= maxepochs; epoch++ {
		tevol := time.Now()
		log.Println("NovEnvs", epoch, ":", popstart.NovEnvs)
		envtraj = append(envtraj, popstart.NovEnvs)
		if epoch != 0 {
			fmt.Println("Epoch ", epoch, "has environments", popstart.NovEnvs)
		}

		pop1 := popstart.Evolve(test_flag, ftraj, json_out, epochlength, epoch)
		fmt.Println("End of epoch", epoch)

		if !test_flag && epoch == maxepochs { //Export output population; just before epoch change
			pop1.ToJSON(json_out)
		}
		dtevol := time.Since(tevol)
		fmt.Println("Time taken to simulate evolution :", dtevol)

		popstart = pop1 //Update population after evolution.
		popstart.ChangeEnvs(denv)
		err = multicell.DeepVec3NovTest(popstart.NovEnvs, envtraj)
		if err != nil {
			fmt.Println(err)
		}
		novvec = append(novvec, err == nil)
	}
	err = ftraj.Close()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Trajectory of population written to %s \n", T_Filename)
	fmt.Printf("JSON encoding of evolved population written to %s \n", jfilename)

	fmt.Println("Novelty of environment cue :", novvec)
	dt := time.Since(t0)
	fmt.Println("Total time taken : ", dt)
}
