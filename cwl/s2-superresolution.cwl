cwlVersion: v1.2
$namespaces:
  s: https://schema.org/
  js: https://json-schema.org/
$schemas:
- http://schema.org/version/latest/schemaorg-current-http.rdf
$graph:
- class: Workflow
  id: main
  label: Antflow SuperResolution
  doc: "Workflow Sentinel2 SuperResolution"

  inputs:
    input:
      type: string
      doc: Path to the input Sentinel2 L2A product directory or zip
    model:
      type: string
      default: "carn_3x3x64g4sw_bootstrap.yaml"
      doc: Name of the yaml file describing the model    
    region_of_interest_pixel:
      type:
      - "null"
      - type: array
        items: float
      doc: Restrict region of interest to process (expressed in in row/col [col_start line_start col_end line_end], with respect to 10 meter pixels)
    region_of_interest:
      type:
      - "null"
      - type: array
        items: float
      doc: Restrict region of interest to process (expressed in utm coordinates [left bottom right top])
    l1c:
      type: ["null", boolean]
      default: False
      doc: Input product is Sentinel2 L1C
    l3a:
      type: ["null", boolean]
      default: False
      doc: Input product is Sentinel2 L3A
    bicubic:
      type: ["null", boolean]
      default: False
      doc: Also generate bicubic upsampled image
    gpu:
      type: ["null", boolean]
      default: False
      doc: Run inference on GPUs if available
    number_of_threads:
      type: ["null", int]
      default: 8
      doc: Number of threads used for model inference

  outputs:
    result:
      type:
        type: array
        items: File
      outputSource: s2_superresolution/result

  steps:
    download_data:
      run: "#download_data"
      in:
        s3_uri: input
      out: [safe_product]
    s2_superresolution:
      run: "#s2_superresolution"
      in:
        s2_product: download_data/safe_product
        model: model
        region_of_interest_pixel: region_of_interest_pixel
        region_of_interest: region_of_interest
        l1c: l1c
        l3a: l3a
        bicubic: bicubic
        gpu: gpu
        number_of_threads: number_of_threads
      out: [result]

- class: CommandLineTool
  id: download_data
  doc: "ETL Step - Download input product from S3 storage."

  requirements:
    InlineJavascriptRequirement: {}
    DockerRequirement:
      dockerPull: ghcr.io/pjtys/aws-cli:2.35.8
    ResourceRequirement:
      coresMin: 1
      coresMax: 1
      ramMin: 256
      ramMax: 512
    NetworkAccess:
      networkAccess: true
    EnvVarRequirement:
      envDef:
        AWS_ENDPOINT_URL: "https://s3.fr-par.scw.cloud"

  inputs:
    s3_uri:
      type: string
  
  baseCommand: aws
  arguments:
    - position: 1
      valueFrom: "s3"
    - position: 2
      valueFrom: "cp"
    - position: 3
      valueFrom: $(inputs.s3_uri)
    - position: 4
      valueFrom: $(inputs.s3_uri.split('/').pop())
    - position: 5
      valueFrom: "--recursive"

  outputs:
    safe_product:
      type: Directory
      outputBinding:
        glob: "*.SAFE"

- class: CommandLineTool
  id: s2_superresolution
  doc: "SuperResolution Step - Execute S2 SuperResolution"

  requirements:
    DockerRequirement:
      dockerPull: ghcr.io/pjtys/s2s:0.0.2
    ResourceRequirement:
      coresMin: 1
      coresMax: 1
      ramMin: 2048
      ramMax: 4096
  
  inputs:
    s2_product:
      type: Directory
    model:
      type: string
    region_of_interest_pixel:
      type:
      - "null"
      - type: array
        items: float
      inputBinding:
        prefix: --region_of_interest_pixel 
    region_of_interest:
      type:
      - "null"
      - type: array
        items: float
      inputBinding:
        prefix: --region_of_interest 
    l1c:
      type: ["null", boolean]
      default: False
      inputBinding:
        prefix: --l1c 
    l3a:
      type: ["null", boolean]
      default: False
      inputBinding:
        prefix: --l3a
    bicubic:
      type: ["null", boolean]
      default: False
      inputBinding:
        prefix: --bicubic
    gpu:
      type: ["null", boolean]
      default: False
      inputBinding:
        prefix: --gpu 
    number_of_threads:
      type: ["null", int]
      default: 8
      inputBinding:
        prefix: --number_of_threads 
  
  baseCommand: python
  arguments: ["/app/run.py", 
    "-i", "$(inputs.s2_product.path)", 
    "-o", "./result",
    "--model", "/app/src/sentinel2_superresolution/models/$(inputs.model)"]

  outputs:
    result:
      type:
        type: array
        items: File
      outputBinding:
        glob: "./result/*"
