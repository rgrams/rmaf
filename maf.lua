-- maf
-- https://github.com/bjornbytes/maf
-- MIT License

local ffi = type(jit) == 'table' and jit.status() and require 'ffi'
local vec3, quat
local sin, cos, acos, sqrt, PI = math.sin, math.cos, math.acos, math.sqrt, math.pi

local forward
local vtmp1
local vtmp2
local qtmp1

local function cross(ax, ay, az, bx, by, bz)
	return ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx
end

vec3 = {
	__call = function(_, x, y, z)
		return setmetatable({ x = x or 0, y = y or x or 0, z = z or x or 0 }, vec3)
	end,

	__tostring = function(v)
		return string.format('(%f, %f, %f)', v.x, v.y, v.z)
	end,

	__add = function(v, u) return v:add(u, vec3()) end,
	__sub = function(v, u) return v:sub(u, vec3()) end,
	__mul = function(v, u)
		if vec3.isvec3(u) then return v:mul(u, vec3())
		elseif type(u) == 'number' then return v:scale(u, vec3())
		else error('vec3s can only be multiplied by vec3s and numbers') end
	end,
	__div = function(v, u)
		if vec3.isvec3(u) then return v:div(u, vec3())
		elseif type(u) == 'number' then return v:scale(1 / u, vec3())
		else error('vec3s can only be divided by vec3s and numbers') end
	end,
	__unm = function(v) return v:scale(-1) end,
	__len = function(v) return v:length() end,

	__index = {
		isvec3 = ffi and function(x)  return ffi.istype('vec3', x)  end or
			function(x)  return getmetatable(x) == vec3  end,

		clone = function(v)
			return vec3(v.x, v.y, v.z)
		end,

		unpack = function(v)
			return v.x, v.y, v.z
		end,

		set = function(v, x, y, z)
			if vec3.isvec3(x) then x, y, z = x.x, x.y, x.z end
			v.x = x
			v.y = y or x
			v.z = z or x
			return v
		end,

		add = function(v, u, out)
			out = out or v
			out.x = v.x + u.x
			out.y = v.y + u.y
			out.z = v.z + u.z
			return out
		end,

		sub = function(v, u, out)
			out = out or v
			out.x = v.x - u.x
			out.y = v.y - u.y
			out.z = v.z - u.z
			return out
		end,

		mul = function(v, u, out)
			out = out or v
			out.x = v.x * u.x
			out.y = v.y * u.y
			out.z = v.z * u.z
			return out
		end,

		div = function(v, u, out)
			out = out or v
			out.x = v.x / u.x
			out.y = v.y / u.y
			out.z = v.z / u.z
			return out
		end,

		scale = function(v, s, out)
			out = out or v
			out.x = v.x * s
			out.y = v.y * s
			out.z = v.z * s
			return out
		end,

		length = function(v)
			return sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
		end,

		len2 = function(v)
			return v.x * v.x + v.y * v.y + v.z * v.z
		end,

		normalize = function(v, out)
			out = out or v
			local len = v:length()
			return len == 0 and v or v:scale(1 / len, out)
		end,

		distance = function(v, u)
			return vec3.sub(v, u, vtmp1):length()
		end,

		angle = function(v, u)
			return acos(v:dot(u) / (v:length() + u:length()))
		end,

		dot = function(v, u)
			return v.x * u.x + v.y * u.y + v.z * u.z
		end,

		cross = function(v, u, out)
			out = out or v
			out.x, out.y, out.z = cross(v.x, v.y, v.z, u.x, u.y, u.z)
			return out
		end,

		lerp = function(v, u, t, out)
			out = out or v
			out.x = v.x + (u.x - v.x) * t
			out.y = v.y + (u.y - v.y) * t
			out.z = v.z + (u.z - v.z) * t
			return out
		end,

		lerpdt = function(v, u, rate, dt, out)
			out = out or v
			local k = (1 - rate)^dt -- Flip rate so it's the expected direction (0 = no change).
			out.x = u.x + (v.x - u.x) * k
			out.y = u.y + (v.y - u.y) * k
			out.z = u.z + (v.z - u.z) * k
			return out
		end,

		project = function(v, u, out)
			out = out or v
			local unorm = vtmp1
			u:normalize(unorm)
			local dot = v:dot(unorm)
			out.x = unorm.x * dot
			out.y = unorm.y * dot
			out.z = unorm.z * dot
			return out
		end,

		rotate = function(v, q, out)
			-- From: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Used_methods
			-- Formula: out = v + 2r × (r × v + w*v)    (r == vector part of quat)
			out = out or v
			-- Localize stuff:
			local vx, vy, vz = v.x, v.y, v.z
			local rx, ry, rz, w = q.x, q.y, q.z, q.w
			-- Compute part in parentheses:
			local rcrossvx, rcrossvy, rcrossvz = cross(rx,ry,rz, vx,vy,vz)
			local _x, _y, _z = rcrossvx + vx*w, rcrossvy + vy*w, rcrossvz + vz*w
			_x, _y, _z = cross(2*rx,2*ry,2*rz, _x,_y,_z)
			-- Put it all together:
			out.x, out.y, out.z = vx + _x, vy + _y, vz + _z
			return out
		end,
	}
}

quat = {
	__call = function(_, x, y, z, w)
		return setmetatable({ x = x or 0, y = y or 0, z = z or 0, w = w or 1 }, quat)
	end,

	__tostring = function(q)
		return string.format('(%f, %f, %f, %f)', q.x, q.y, q.z, q.w)
	end,

	__add = function(q, r) return q:add(r, quat()) end,
	__sub = function(q, r) return q:sub(r, quat()) end,
	__mul = function(q, r)
		if quat.isquat(r) then return q:mul(r, quat())
		elseif vec3.isvec3(r) then return r:rotate(q, vec3())
		else error('quats can only be multiplied by quats and vec3s') end
	end,
	__unm = function(q) return q:scale(-1) end,
	__len = function(q) return q:length() end,

	__index = {
		isquat = ffi and function(x)  return ffi.istype('quat', x)  end or
			function(x)  return getmetatable(x) == quat  end,

		clone = function(q)
			return quat(q.x, q.y, q.z, q.w)
		end,

		unpack = function(q)
			return q.x, q.y, q.z, q.w
		end,

		set = function(q, x, y, z, w)
			if quat.isquat(x) then x, y, z, w = x.x, x.y, x.z, x.w end
			q.x = x
			q.y = y
			q.z = z
			q.w = w
			return q
		end,

		fromX = function(angle, out)
			out = out or quat()
			out.x, out.y, out.z = 1 * sin(angle * 0.5), 0, 0
			out.w = cos(angle * 0.5)
			return out
		end,

		fromY = function(angle, out)
			out = out or quat()
			out.x, out.y, out.z = 0, 1 * sin(angle * 0.5), 0
			out.w = cos(angle * 0.5)
			return out
		end,

		fromZ = function(angle, out)
			out = out or quat()
			out.x, out.y, out.z = 0, 0, 1 * sin(angle * 0.5)
			out.w = cos(angle * 0.5)
			return out
		end,

		fromAngleAxis = function(angle, x, y, z)
			return quat():setAngleAxis(angle, x, y, z)
		end,

		fromMatrix = function(m, out)
			out = out or quat()
			local m11, m12, m13 = m[1], m[5], m[ 9]
			local m21, m22, m23 = m[2], m[6], m[10]
			local m31, m32, m33 = m[3], m[7], m[11]
			local t, x, y, z, w
			if m33 < 0 then
				if m11 > m22 then
					t = 2 + m11 - m22 - m33
					x, y, z, w = t, m12+m21, m31+m13, m23-m32
				else
					t = 2 - m11 + m22 - m33
					x, y, z, w = m12+m21, t, m23+m32, m31-m13
				end
			else
				if (m11 < -m22) then
					t = 2 - m11 - m22 + m33
					x, y, z, w = m31+m13, m23+m32, t, m12-m21
				else
					t = 2 + m11 + m22 + m33
					x, y, z, w = m23-m32, m31-m13, m12-m21, t
				end
			end
			local k = 0.5/sqrt(t)
			out.x, out.y, out.z, out.w = x*k, y*k, z*k, w*k
			return out
		end,

		setAngleAxis = function(q, angle, x, y, z)
			if vec3.isvec3(x) then x, y, z = x.x, x.y, x.z end
			local s = sin(angle * .5)
			local c = cos(angle * .5)
			q.x = x * s
			q.y = y * s
			q.z = z * s
			q.w = c
			return q
		end,

		getAngleAxis = function(q)
			if q.w > 1 or q.w < -1 then q:normalize() end
			local s = sqrt(1 - q.w * q.w)
			s = s < .0001 and 1 or 1 / s
			return 2 * acos(q.w), q.x * s, q.y * s, q.z * s
		end,

		between = function(u, v)
			return quat():setBetween(u, v)
		end,

		setBetween = function(q, u, v)
			local dot = u:dot(v)
			if dot > .99999 then
				q.x, q.y, q.z, q.w = 0, 0, 0, 1
				return q
			elseif dot < -.99999 then
				vtmp1.x, vtmp1.y, vtmp1.z = 1, 0, 0
				vtmp1:cross(u)
				if #vtmp1 < .00001 then
					vtmp1.x, vtmp1.y, vtmp1.z = 0, 1, 0
					vtmp1:cross(u)
				end
				vtmp1:normalize()
				return q:setAngleAxis(PI, vtmp1)
			end

			q.x, q.y, q.z = u.x, u.y, u.z
			vec3.cross(q, v)
			q.w = 1 + dot
			return q:normalize()
		end,

		fromDirection = function(x, y, z)
			return quat():setDirection(x, y, z)
		end,

		setDirection = function(q, x, y, z)
			if vec3.isvec3(x) then x, y, z = x.x, x.y, x.z end
			vtmp2.x, vtmp2.y, vtmp2.z = x, y, z
			return q:setBetween(forward, vtmp2)
		end,

		add = function(q, r, out)
			out = out or q
			out.x = q.x + r.x
			out.y = q.y + r.y
			out.z = q.z + r.z
			out.w = q.w + r.w
			return out
		end,

		sub = function(q, r, out)
			out = out or q
			out.x = q.x - r.x
			out.y = q.y - r.y
			out.z = q.z - r.z
			out.w = q.w - r.w
			return out
		end,

		mul = function(q, r, out)
			out = out or q
			local qx, qy, qz, qw = q:unpack()
			local rx, ry, rz, rw = r:unpack()
			out.x = qx * rw + qw * rx + qy * rz - qz * ry
			out.y = qy * rw + qw * ry + qz * rx - qx * rz
			out.z = qz * rw + qw * rz + qx * ry - qy * rx
			out.w = qw * rw - qx * rx - qy * ry - qz * rz
			return out
		end,

		scale = function(q, s, out)
			out = out or q
			out.x = q.x * s
			out.y = q.y * s
			out.z = q.z * s
			out.w = q.w * s
			return out
		end,

		length = function(q)
			return sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
		end,

		reverse = function(q, out) -- The conjugate.
			out = out or q
			out.x, out.y, out.z, out.w = -q.x, -q.y, -q.z, q.w
			return out
		end,

		normalize = function(q, out)
			out = out or q
			local len = q:length()
			return len == 0 and q or q:scale(1 / len, out)
		end,

		lerp = function(q, r, t, out)
			out = out or q
			r:scale(t, qtmp1)
			q:scale(1 - t, out)
			return out:add(qtmp1)
		end,

		slerp = function(q, r, t, out)
			out = out or q

			local dot = q.x * r.x + q.y * r.y + q.z * r.z + q.w * r.w
			if dot < 0 then
				dot = -dot
				r:scale(-1)
			end

			if 1 - dot < .0001 then
				return q:lerp(r, t, out)
			end

			local theta = acos(dot)
			q:scale(sin((1 - t) * theta), out)
			r:scale(sin(t * theta), qtmp1)
			return out:add(qtmp1):scale(1 / sin(theta))
		end
	}
}

-- Don't want __tostring or operator methods on the class itself.
local vec3_mt = { __call = vec3.__call, __index = vec3.__index }
local quat_mt = { __call = quat.__call, __index = quat.__index }

if ffi then
	ffi.cdef [[
	typedef struct { double x, y, z; } vec3;
	typedef struct { double x, y, z, w; } quat;
	]]
	-- FFI structs -do not- use the __call method, so we must keep the modules as lua tables.
	local new_vec3 = ffi.typeof('vec3')
	local new_quat = ffi.typeof('quat')
	vec3.new = new_vec3 -- More efficient.
	quat.new = new_quat -- More efficient, if you don't mind the default w=0.
	vec3_mt.__call = function(_, x, y, z)
		return new_vec3(x or 0, y or x or 0, z or x or 0)
	end
	quat_mt.__call = function(_, x, y, z, w)
		return new_quat(x or 0, y or 0, z or 0, w or 1)
	end
	ffi.metatype('vec3', vec3)
	ffi.metatype('quat', quat)
end

setmetatable(vec3, vec3_mt)
setmetatable(quat, quat_mt)

forward = vec3(0, 0, -1)
vtmp1 = vec3()
vtmp2 = vec3()
qtmp1 = quat()

return {
	vec3 = vec3,
	quat = quat,

	vector = vec3,
	rotation = quat
}
