# MethyCNV
The code and manual used in MethyCNV project

1. download aligned BAM file from GDC data portal
2. run CNVnator (v.0.3.3). Only focus on Chr 1-22, exclude Chr X and Y.

```
module load cnvnator/0.3.3
cnvnator  -root $temp_dir/root.file  -genome  $genome_fasta  -unique  -tree $raw_dir/*/*.bam

cnvnator -genome $genome_fasta -root $temp_dir/root.file -his 100 -d $split_chrom

cnvnator -root $temp_dir/root.file -chrom 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22  -stat 100

cnvnator -root $temp_dir/root.file -chrom 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22  -partition 100

cnvnator -root $temp_dir/root.file -chrom 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22  -call 100 |& tee $temp_dir/$sample".calls.txt"

rm $temp_dir/root.file
```

2. call SNV and indel using GATK best practice. 

The genome was splitted into many intervals to fast the speed.
```
module load gatk/3.7
java -jar /risapps/rhel7/gatk/3.7/GenomeAnalysisTK.jar \
-T HaplotypeCaller \
-R $genome_fasta \
-I $raw_dir/*/*.bam \
-o $temp_dir/SNP_dir/$list_name.raw.vcf \
-ERC GVCF \
-L $interval_dir/$list_file \
--variant_index_type LINEAR \
--variant_index_parameter 128000 
```
Combine the pieces of results
```
java -cp  /risapps/rhel7/gatk/3.7/GenomeAnalysisTK.jar org.broadinstitute.gatk.tools.CatVariants -R  $genome_fasta -V $temp_dir/SNP_dir/$file_name -out  $temp_dir/SNP.raw.vcf
```
Create gvcf file based on the raw SNP+indel calls
```
java -jar /risapps/rhel7/gatk/3.7/GenomeAnalysisTK.jar \
   -T GenotypeGVCFs  \
   -R $genome_fasta  \
   --variant $temp_dir/SNP.raw.vcf \
   -o $temp_dir/SNP.raw2.vcf \
```
Reclibration for SNP and indel separately  
```
module load gatk/3.7
module load R/3.5.0   

java -jar /risapps/rhel7/gatk/3.7/GenomeAnalysisTK.jar \
   -T VariantRecalibrator \
   -R $genome_fasta  \
   -input $temp_dir/SNP.raw2.vcf \
   -recalFile $temp_dir/recalibrate.snp.recal \
   -tranchesFile $temp_dir/recalibrate.snp.tranches \
   -nt 1 \
   -resource:hapmap,known=false,training=true,truth=true,prior=15.0 $GATK_resource/hapmap_3.3.b37.vcf \
   -resource:omni,known=false,training=true,truth=true,prior=12.0 $GATK_resource/1000G_omni2.5.b37.vcf \
   -resource:1000G,known=false,training=true,truth=false,prior=10.0 $GATK_resource/1000G_phase1.snps.high_confidence.b37.vcf \
   -resource:dbsnp,known=true,training=false,truth=false,prior=2.0 $GATK_resource/dbsnp_138.b37.vcf \
   -an QD \
   -an MQ \
   -an MQRankSum \
   -an ReadPosRankSum \
   -an FS \
   -an SOR \
   -an DP \
   -mode SNP
   

java -jar /risapps/rhel7/gatk/3.7/GenomeAnalysisTK.jar \
   -T ApplyRecalibration \
   -R $genome_fasta \
   -input $temp_dir/SNP.raw2.vcf \
   -tranchesFile $temp_dir/recalibrate.snp.tranches \
   -recalFile $temp_dir/recalibrate.snp.recal \
   -o $temp_dir/recalibrated_snp_raw_indel.vcf  \
   --ts_filter_level 99.5 \
   -mode SNP


java -jar /risapps/rhel7/gatk/3.7/GenomeAnalysisTK.jar \
   -T VariantRecalibrator \
   -R $genome_fasta  \
   -input $temp_dir/recalibrated_snp_raw_indel.vcf \
   -recalFile $temp_dir/recalibrate.indel.recal \
   -tranchesFile $temp_dir/recalibrate.indel.tranches \
   -nt 1 \
   --maxGaussians 4 \
   -resource:mills,known=false,training=true,truth=true,prior=12.0 $GATK_resource/Mills_and_1000G_gold_standard.indels.b37.vcf  \
   -resource:dbsnp,known=true,training=false,truth=false,prior=2.0 $GATK_resource/dbsnp_138.b37.vcf \
   -an QD    -an DP -an FS -an SOR -an ReadPosRankSum -an MQRankSum  \
   -mode INDEL   

java -jar /risapps/rhel7/gatk/3.7/GenomeAnalysisTK.jar \
   -T ApplyRecalibration \
   -R $genome_fasta \
   -input $temp_dir/recalibrated_snp_raw_indel.vcf \
   -tranchesFile $temp_dir/recalibrate.indel.tranches \
   -recalFile $temp_dir/recalibrate.indel.recal \
   -o $temp_dir/recalibrated_output.vcf  \
   --ts_filter_level 99.0 \
   -mode INDEL
```
3. Use the reclibrated SNP+indel data to call ERDS
```
module load perl/5.24.1
module load python/2.7.15-anaconda

perl $erds_dir/erds_pipeline.pl -b $raw_dir/*/*.bam -v $temp_dir/recalibrated_output.vcf -o $temp_dir/  \
-r $genome_fasta -sd b37
```
4. Tune the final CNV calls

merge ERDS and CNVnator data to erds+. Code adapted from "TCAG-WGS-CNV-workflow/process_cnvs.erds+.sh".
```
$script_dir/generate_erds+.sh $pipeline_dir $temp_dir/*.erds.vcf  $temp_dir/*.calls.txt  $temp_dir/temp  /risapps/rhel7/python/2.7.13/bin/python2.7
```
filter erds+ file, keep all duplication detected by ERDS+ to increase sensitivity. Keep only intersected duplicated CNVs by CNVnator and ERDS.
```
python $script_dir/filter_erds+file.py -i $temp_dir/temp/*.erds+.txt -o $temp_dir/temp/$sample".erds+.filtered.txt"
```
overlapping erds+ with the RLCR data set. $RLCR can be full RLCR or non RepeatMasker RLCR (use this to increase sensitivity).
```
python $pipeline_dir/compare_with_RLCR_definition.py  $RLCR   $temp_dir/temp/$sample".erds+.filtered.txt"
```
remove CNVs with >=70% overlapping with RLCRs.
```
python $script_dir/filter_erds+.RLCR.py -i  $temp_dir/temp/$sample".erds+.filtered.txt.RLCR"  -o $temp_dir/$sample".erds+.filter.RLCR.filter.txt"
```



##How to run each commands.
#with Ten new folders with the manifest file moved to the By_portionID folder
```
sh 1_write_setup.sh 
sh 1.5_run_setup.sh
sh run_setup.sh
```
0. Download data from the GDC for each sample
```
module load python/2.7.13-gdc 
gdc-client  download -m ./k450_WGS_download_manifest.txt  -t /rsrch3/scratch/radonc-rsrch/jyang32/MethyCNV/token/gdc-user-token.2018-12-17T02_07_53.251Z.txt
```

1. First run all .sh and .py files in code file
```
sh section1.sh
python 3.5_submit_call_snp.py

```
2. Run section1
```
sh submit_call_SNP.sh

```
3. Run section2.
Check SNP files generated and CNVnator file.
```
ls|grep "idx" |wc -l  ## need to be 220
```
If yes. proceed to create section2.lsf
```
sh section2.sh
bsub < section2.sh

```
4. Run the IDAT file part
```
sh section3.sh
bsub < call_IDAT.lsf
```
5. Then all final data will be in final_dir, copy them out
```
cp ../By_portionID/*/finaldata/* ./
```



