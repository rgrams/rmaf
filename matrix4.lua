
-- Copyright Ross Grams 2024 - MIT License

-- 4x4 Transformation Matrix Math Library.
-- Row-major matrices: Vectors are on rows, position is in the column on the right.
-- Uses C structs if FFI is available.

local ffi = type(jit) == 'table' and jit.status() and require 'ffi'

local folder = (...):gsub("matrix4$", "")
local vec3 = require(folder.."maf").vec3
local quat = require(folder.."maf").quat

local matrix, matrix_mt

local function set(M, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16)
	M[1], M[2], M[3], M[4] = _1, _2, _3, _4
	M[5], M[6], M[7], M[8] = _5, _6, _7, _8
	M[9], M[10], M[11], M[12] = _9, _10, _11, _12
	M[13], M[14], M[15], M[16] = _13, _14, _15, _16
	return M
end

local function copy(from, to) -- Tested to be exactly as fast as set().
	for i=1,16 do
		to[i] = from[i]
	end
	return to
end

local function _unpack(M) -- About twice as fast as using `unpack()`.
	return M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8], M[9], M[10], M[11], M[12], M[13], M[14], M[15], M[16]
end

local function new()
	return setmetatable({}, matrix_mt)
end

local function identity(M)
	if M then  return copy(matrix.IDENTITY, M)  end
	return set(new(),
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	)
end

local _printFormat = [[[matrix] {
	%.2f, %.2f, %.2f, %.2f,
	%.2f, %.2f, %.2f, %.2f,
	%.2f, %.2f, %.2f, %.2f,
	%.2f, %.2f, %.2f, %.2f,
}]]

local abs = math.abs
local function isZero(x)
	return abs(x) < 0.000001
end

-- (For invert.) Get determinant of 3x3 matrix (AKA: the minor of a 4x4 matrix element).
local function det3(a, b, c, d, e, f, g, h, i)
	return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
end

-- Use one row of the cofactor matrix to calculate the determinant.
local function determinant(M)
	local a, b, c, d = M[1], M[2], M[3], M[4]
	local e, f, g, h = M[5], M[6], M[7], M[8]
	local i, j, k, l = M[9], M[10], M[11], M[12]
	local m, n, o, p = M[13], M[14], M[15], M[16]
	local inv_1  =  det3(f,g,h, j,k,l, n,o,p)
	local inv_5  = -det3(e,g,h, i,k,l, m,o,p)
	local inv_9  =  det3(e,f,h, i,j,l, m,n,p)
	local inv_13 = -det3(e,f,g, i,j,k, m,n,o)
	return a*inv_1 + b*inv_5 + c*inv_9 + d*inv_13
end

local function fromQuat(q)
	local qx, qy, qz, qw = q.x, q.y, q.z, q.w
	local qx2, qy2, qz2 = qx*qx, qy*qy, qz*qz

	local xx, xy, xz = 1 - 2*qy2 - 2*qz2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw
	local yx, yy, yz = 2*qx*qy + 2*qz*qw, 1 - 2*qx2 - 2*qz2, 2*qy*qz - 2*qx*qw
	local zx, zy, zz = 2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx2 - 2*qy2
	return xx, xy, xz, 0,
	       yx, yy, yz, 0,
	       zx, zy, zz, 0,
			  0,  0,  0, 1
end

-- Some Vec3 functions needed for lookAt() (that don't create extra tables/objects).
local sqrt = math.sqrt

local function lenv3(x, y, z)
	return sqrt(x*x + y*y + z*z)
end

local function normalizev3(x, y, z)
	local len = lenv3(x, y, z)
	if len == 0 then  return x, y, z  end
	return x/len, y/len, z/len
end

local function crossv3(x1, y1, z1, x2, y2, z2)
	local x = y1*z2 - z1*y2
	local y = z1*x2 - x1*z2
	local z = x1*y2 - y1*x2
	return x, y, z
end

local function dotv3(x1, y1, z1, x2, y2, z2)
	return x1*x2, y1*y2, z1*z2
end

local function __call(_, a, b, ...)
	if a then
		if b and tonumber(b) then
			return set(new(), a, b, ...)
		elseif matrix.ismat4(a) then
			return copy(a, new())
		end
	else
		return identity()
	end
end

local tmp = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

matrix_mt = {
	__tostring = function(M)  return _printFormat:format(_unpack(M))  end,
	__mul = function(M, x)
		if vec3.isvec3(x) then  return M:vmult(x, 1, vec3())
		elseif matrix.ismat4(x) then  return M:mult(x, new())
		else  error('matrix4s can only be multiplied by vec3s and matrix4s')  end
   end,
	__index = {
		LAYOUT = "row",
		ismat4 = function(M)
			return getmetatable(M) == matrix_mt
		end,
		set = set,
		copy = function(M, out)
			return copy(M, out or new())
		end,
		identity = identity,
		unpack = _unpack,
		getPos = function(m)
			return m[4], m[8], m[12]
		end,
		setPos = function(m, x, y, z)
			m[4], m[8], m[12] = x, y, z
		end,
		fromTransform = function(pos, rot, scale, out)
			out = out or new()
			local sx, sy, sz = scale.x, scale.y, scale.z
			local xx, xy, xz, _,
			      yx, yy, yz, _,
			      zx, zy, zz, _ = fromQuat(rot)
			set(out, -- Needs to be rot * scale, not the other way around.
				xx*sx, xy*sy, xz*sz, pos.x,
				yx*sx, yy*sy, yz*sz, pos.y,
				zx*sx, zy*sy, zz*sz, pos.z,
				    0,     0,     0,     1
			)
			return out
		end,
		fromQuat = function(q, out)
			return set(out or new(), fromQuat(q))
		end,
		decompose = function(m, outPos, outRot, outScale)
			outPos = outPos or vec3()
			outRot = outRot or quat()
			outScale = outScale or vec3()
			-- Position
			outPos:set(m[4], m[8], m[12])
			-- Scale
			local sx = lenv3(m[1], m[5], m[9])
			local sy = lenv3(m[2], m[6], m[10])
			local sz = lenv3(m[3], m[7], m[11])
			outScale:set(sx, sy, sz)
			-- Rotation
			set(tmp,
				m[1]/sx, m[5]/sx,  m[9]/sx,  0,
				m[2]/sy, m[6]/sy,  m[10]/sy,  0,
				m[3]/sz, m[7]/sz,  m[11]/sz, 0,
				0,       0,        0,        0
			)
			quat.fromMatrix(tmp, outRot)
			return outPos, outRot, outScale
		end,
		orthographicSym = function(width, height, near, far, out) -- Symmetrical x and y.
			local rt, top = width/2, height/2
			return set(out or new(),
				1/rt,   0,       0,           0,
				0,      1/top,   0,           0,
				0,      0,      -2/far-near, -(far+near)/(far-near),
				0,      0,       0,           1
			)
		end,
		perspective = function(fovy, near, far, aspect, out)
			local invHyp = 1/math.tan(fovy/2)
			return set(out or new(),
				invHyp/aspect, 0,         0,                      0,
				0,             invHyp,    0,                      0,
				0,             0,        -(far+near)/(far-near), -2*far*near/(far-near),
				0,             0,        -1,                      0
			)
		end,
		lookAt = function(eye, target, up, out)
			local zx, zy, zz = normalizev3(eye.x - target.x, eye.y - target.y, eye.z - target.z)
			local xx, xy, xz = normalizev3(crossv3(up.x, up.y, up.z, zx, zy, zz))
			local yx, yy, yz = crossv3(zx, zy, zz, xx, xy, xz)
			return set(out or new(),
				xx, xy, xz, -dotv3(xx, xy, xz, eye.x, eye.y, eye.z),
				yx, yy, yz, -dotv3(yx, yy, yz, eye.x, eye.y, eye.z),
				zx, zy, zz, -dotv3(zx, zy, zz, eye.x, eye.y, eye.z),
				 0,  0,  0,  1
			)
		end,
		transpose = function (a, out)
			out = out or a
			local _1,  _2,  _3,  _4  = a[1],  a[2],  a[3],  a[4]
			local _5,  _6,  _7,  _8  = a[5],  a[6],  a[7],  a[8]
			local _9,  _10, _11, _12 = a[9],  a[10], a[11], a[12]
			local _13, _14, _15, _16 = a[13], a[14], a[15], a[16]
			out[1], out[5], out[9],  out[13] = _1,  _2,  _3,  _4
			out[2], out[6], out[10], out[14] = _5,  _6,  _7,  _8
			out[3], out[7], out[11], out[15] = _9,  _10, _11, _12
			out[4], out[8], out[12], out[16] = _13, _14, _15, _16
			return out
		end,
		mult = function(a, b, out)
			out = out or new()
			-- Save values in case `out` is `a` or `b`.
			local a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16 = _unpack(a)
			local b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16 = _unpack(b)
			-- Rows of `a` multiplied by columns of `b`.
			out[1] = a1*b1 + a2*b5 + a3*b9 + a4*b13
			out[2] = a1*b2 + a2*b6 + a3*b10 + a4*b14
			out[3] = a1*b3 + a2*b7 + a3*b11 + a4*b15
			out[4] = a1*b4 + a2*b8 + a3*b12 + a4*b16
			out[5] = a5*b1 + a6*b5 + a7*b9 + a8*b13
			out[6] = a5*b2 + a6*b6 + a7*b10 + a8*b14
			out[7] = a5*b3 + a6*b7 + a7*b11 + a8*b15
			out[8] = a5*b4 + a6*b8 + a7*b12 + a8*b16
			out[9] = a9*b1 + a10*b5 + a11*b9 + a12*b13
			out[10] = a9*b2 + a10*b6 + a11*b10 + a12*b14
			out[11] = a9*b3 + a10*b7 + a11*b11 + a12*b15
			out[12] = a9*b4 + a10*b8 + a11*b12 + a12*b16
			out[13] = a13*b1 + a14*b5 + a15*b9 + a16*b13
			out[14] = a13*b2 + a14*b6 + a15*b10 + a16*b14
			out[15] = a13*b3 + a14*b7 + a15*b11 + a16*b15
			out[16] = a13*b4 + a14*b8 + a15*b12 + a16*b16
			return out
		end,
		vmult = function(M, v, w, out)
			-- Just like matrix-matrix mult., multiply rows of `m` by columns (only one) of `v`.
			-- Equals the dot product (element-wise multiplication) of each row with the vector's axis values.
			--[[ -- For reference:
			[ 1,  2,  3,  4,  ]    [ x ]
			| 5,  6,  7,  8,  |    | y |
			| 9,  10, 11, 12, |    | z |
			[ 13, 14, 15, 16  ]    [ w ]  --]]
			w = w or 1 -- Transforms a -position- by default, not a vector.
			out = out or vec3()
			local x, y, z = v.x, v.y, v.z
			out.x = M[1]*x + M[2]*y + M[3]*z + M[4]*w
			out.y = M[5]*x + M[6]*y + M[7]*z + M[8]*w
			out.z = M[9]*x + M[10]*y + M[11]*z + M[12]*w
			local outw = M[13]*x + M[14]*y + M[15]*z + M[16]*w
			return out, outw
		end,
		invert = function(M, out)
			local a, b, c, d = M[1], M[2], M[3], M[4]
			local e, f, g, h = M[5], M[6], M[7], M[8]
			local i, j, k, l = M[9], M[10], M[11], M[12]
			local m, n, o, p = M[13], M[14], M[15], M[16]
			local inv = out or {}
			-- Cofactor matrix is transposed matrix of minors with alternating signs.
			-- One row or column of cofactor matrix lets you calculate the determinant.
			inv[1]  =  det3(f,g,h, j,k,l, n,o,p) -- Place into transposed indices (row into column)
			inv[5]  = -det3(e,g,h, i,k,l, m,o,p)
			inv[9]  =  det3(e,f,h, i,j,l, m,n,p)
			inv[13] = -det3(e,f,g, i,j,k, m,n,o)
			local det = a*inv[1] + b*inv[5] + c*inv[9] + d*inv[13]
			if isZero(det) then return false end -- Non-invertable, exit early.
			-- 1/det * cofactor-matrix = inverse matrix.
			local scale = 1/det
			inv[1]  = scale*inv[1]
			inv[5]  = scale*inv[5]
			inv[9]  = scale*inv[9]
			inv[13] = scale*inv[13]

			inv[2]  = -scale*det3(b,c,d, j,k,l, n,o,p)
			inv[6]  =  scale*det3(a,c,d, i,k,l, m,o,p)
			inv[10] = -scale*det3(a,b,d, i,j,l, m,n,p)
			inv[14] =  scale*det3(a,b,c, i,j,k, m,n,o)

			inv[3]  =  scale*det3(b,c,d, f,g,h, n,o,p)
			inv[7]  = -scale*det3(a,c,d, e,g,h, m,o,p)
			inv[11] =  scale*det3(a,b,d, e,f,h, m,n,p)
			inv[15] = -scale*det3(a,b,c, e,f,g, m,n,o)

			inv[4]  = -scale*det3(b,c,d, f,g,h, j,k,l)
			inv[8]  =  scale*det3(a,c,d, e,g,h, i,k,l)
			inv[12] = -scale*det3(a,b,d, e,f,h, i,j,l)
			inv[16] =  scale*det3(a,b,c, e,f,g, i,j,k)

			return inv
		end,
		determinant = determinant
	}
}

matrix = setmetatable(matrix_mt.__index, { __call = __call })

if jit then -- Add pointer address (which doesn't work with vanilla Lua) to print format.
	local i = ("[matrix"):len()
	_printFormat = _printFormat:sub(1, i).." %p".._printFormat:sub(i+1, -1)
	matrix_mt.__tostring = function(M)  return _printFormat:format(M, _unpack(M))  end
end

if ffi then
	-- Can't set metamethods on a C array, so use a struct with an array inside.
	ffi.cdef [[
		typedef struct { double _m[17]; bool isDirty; } matrix4;
	]]
	-- Localize metamethods table to use in function we're overwriting it with.
	local __index = matrix_mt.__index
	-- Use indexing metamethods to make it behave the same as the lua table version.
	matrix_mt.__index = function(M, k)
		if type(k) == "number" then
			return M._m[k]
		elseif k == "isDirty" then
			return M.isDirty
		else
			return rawget(__index, k) -- For using named metamethods.
		end
	end
	matrix_mt.__newindex = function(M, k, v)
		if type(k) == "number" then
			M._m[k] = v
		elseif k == "isDirty" then
			M.isDirty = v
		else
			error("Can't set value on cdata matrix with key '"..tostring(k).."'. Only indices from 1-16 and the 'isDirty' key are valid.")
		end
	end
	__index.ismat4 = function(M)
		return ffi.istype("matrix4", M)
	end
	new = ffi.metatype("matrix4", matrix_mt)
end

matrix.new = new
matrix.IDENTITY = identity()

return matrix
