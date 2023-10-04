# Let the input file path be the first argument from console
$filePath = $args[0]
# Let the output file path be the second argument from console
$outFilePath = $args[1]

# Output current working directory for debugging
Write-Host "Current working directory: $pwd"

# Output two paths for debugging
Write-Host "Input file path: $filePath"
Write-Host "Output file path: $outFilePath"

# Extract file name from the path
$fileName = $filePath

# Convert input file name into legal variable name, replacing dots to underscores
$varName = $fileName.Replace(".", "_").Replace("/","_").Replace("\\","_")
# Output variable name for debugging
Write-Host "Variable name: $varName"

# Define length variable name
$varLengthName = $varName + "_len"


$fileContent = [System.IO.File]::ReadAllBytes($filePath)
$cppArray = ""
for ($i = 0; $i -lt $fileContent.Length; $i++) {
    $cppArray += ("0x" + $fileContent[$i].ToString("x2") + ", ")
}
$cppArray = $cppArray.TrimEnd(", ")
$cppLength = $fileContent.Length
[System.IO.File]::WriteAllText($outFilePath, "unsigned char $varName[] = { $cppArray };`nunsigned int $varLengthName = $cppLength;")
