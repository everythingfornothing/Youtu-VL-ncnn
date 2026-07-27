const fs = require("fs");

function readNpy(path) {
  const buffer = fs.readFileSync(path);
  if (buffer.subarray(0, 6).toString("binary") !== "\x93NUMPY") {
    throw new Error(`${path}: invalid NPY magic`);
  }
  const major = buffer[6];
  let headerStart;
  let headerLength;
  if (major === 1) {
    headerStart = 10;
    headerLength = buffer.readUInt16LE(8);
  } else if (major === 2 || major === 3) {
    headerStart = 12;
    headerLength = buffer.readUInt32LE(8);
  } else {
    throw new Error(`${path}: unsupported NPY version ${major}`);
  }
  const header = buffer
    .subarray(headerStart, headerStart + headerLength)
    .toString("latin1");
  const dtypeMatch = header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/);
  const shapeMatch = header.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/);
  if (!dtypeMatch || !shapeMatch) {
    throw new Error(`${path}: malformed NPY header`);
  }
  const shape = shapeMatch[1]
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)
    .map(Number);
  return {
    path,
    buffer,
    dtype: dtypeMatch[1],
    shape,
    dataOffset: headerStart + headerLength,
  };
}

function itemSize(dtype) {
  if (dtype.endsWith("f4") || dtype.endsWith("i4")) {
    return 4;
  }
  if (dtype.endsWith("i8")) {
    return 8;
  }
  throw new Error(`unsupported dtype ${dtype}`);
}

function valueAt(array, index) {
  const offset = array.dataOffset + index * itemSize(array.dtype);
  if (array.dtype.endsWith("f4")) {
    return array.buffer.readFloatLE(offset);
  }
  if (array.dtype.endsWith("i4")) {
    return array.buffer.readInt32LE(offset);
  }
  if (array.dtype.endsWith("i8")) {
    return array.buffer.readBigInt64LE(offset);
  }
  throw new Error(`unsupported dtype ${array.dtype}`);
}

function compare(leftPath, rightPath) {
  const left = readNpy(leftPath);
  const right = readNpy(rightPath);
  if (
    left.dtype !== right.dtype ||
    JSON.stringify(left.shape) !== JSON.stringify(right.shape)
  ) {
    throw new Error("dtype or shape mismatch");
  }
  const count = left.shape.reduce((product, dimension) => product * dimension, 1);
  let differentElements = 0;
  let maxAbsoluteDifference = 0;
  let absoluteDifferenceSum = 0;
  const firstDifferences = [];
  for (let index = 0; index < count; ++index) {
    const leftValue = valueAt(left, index);
    const rightValue = valueAt(right, index);
    if (leftValue !== rightValue) {
      ++differentElements;
      if (firstDifferences.length < 10) {
        firstDifferences.push({
          index,
          left: leftValue.toString(),
          right: rightValue.toString(),
        });
      }
      if (typeof leftValue === "number") {
        const difference = Math.abs(leftValue - rightValue);
        maxAbsoluteDifference = Math.max(maxAbsoluteDifference, difference);
        absoluteDifferenceSum += difference;
      }
    }
  }
  return {
    left: leftPath,
    right: rightPath,
    dtype: left.dtype,
    shape: left.shape,
    elements: count,
    differentElements,
    maxAbsoluteDifference,
    meanAbsoluteDifference: absoluteDifferenceSum / count,
    firstDifferences,
    byteIdentical: left.buffer.equals(right.buffer),
  };
}

if (process.argv.length !== 4) {
  console.error("Usage: node compare_npy.js LEFT.npy RIGHT.npy");
  process.exit(2);
}

console.log(JSON.stringify(compare(process.argv[2], process.argv[3]), null, 2));
