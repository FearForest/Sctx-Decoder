import io
import zstandard
from enum import IntEnum
from io import BytesIO
import texture2ddecoder
import sys
import os
from PIL import Image
import struct
import logging
from multiprocessing import Pool, cpu_count
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BinaryReader(BytesIO):
    def __init__(self, InitialBytes: bytes) -> None:
        super().__init__(InitialBytes)

    def Skip(self, Size: int):
        self.read(Size)

    def ReadBool(self):
        return self.ReadUchar() >= 1

    def ReadChar(self):
        return int.from_bytes(self.read(1), "little", signed=True)

    def ReadUchar(self):
        return int.from_bytes(self.read(1), "little", signed=False)

    def ReadShort(self):
        return int.from_bytes(self.read(2), "little", signed=True)

    def ReadUshort(self):
        return int.from_bytes(self.read(2), "little", signed=False)

    def ReadInt(self):
        return int.from_bytes(self.read(4), "little", signed=True)
    
    def ReadUint(self):
        return int.from_bytes(self.read(4), "little", signed=False)

    def ReadAscii(self):
        Size = self.ReadUchar()
        if Size != 0xFF:
            return self.read(Size).decode('utf8')
        return None

    def ReadTwip(self):
        return self.ReadInt() / 20


class ScPixel(IntEnum):
    EAC_R11 = 170
    EAC_SIGNED_R11 = 172
    EAC_RG11 = 174
    EAC_SIGNED_RG11 = 176
    ETC2_EAC_RGBA8 = 178
    ETC2_EAC_SRGBA8 = 179
    ETC2_RGB8 = 180
    ETC2_SRGB8 = 181
    ETC2_RGB8_PUNCHTHROUGH_ALPHA1 = 182
    ETC2_SRGB8_PUNCHTHROUGH_ALPHA1 = 183
    ASTC_SRGBA8_4x4 = 186
    ASTC_SRGBA8_5x4 = 187
    ASTC_SRGBA8_5x5 = 188
    ASTC_SRGBA8_6x5 = 189
    ASTC_SRGBA8_6x6 = 190
    ASTC_SRGBA8_8x5 = 192
    ASTC_SRGBA8_8x6 = 193
    ASTC_SRGBA8_8x8 = 194
    ASTC_SRGBA8_10x5 = 195
    ASTC_SRGBA8_10x6 = 196
    ASTC_SRGBA8_10x8 = 197
    ASTC_SRGBA8_10x10 = 198
    ASTC_SRGBA8_12x10 = 199
    ASTC_SRGBA8_12x12 = 200
    ASTC_RGBA8_4x4 = 204
    ASTC_RGBA8_5x4 = 205
    ASTC_RGBA8_5x5 = 206
    ASTC_RGBA8_6x5 = 207
    ASTC_RGBA8_6x6 = 208
    ASTC_RGBA8_8x5 = 210
    ASTC_RGBA8_8x6 = 211
    ASTC_RGBA8_8x8 = 212
    ASTC_RGBA8_10x5 = 213
    ASTC_RGBA8_10x6 = 214
    ASTC_RGBA8_10x8 = 215
    ASTC_RGBA8_10x10 = 216
    ASTC_RGBA8_12x10 = 217
    ASTC_RGBA8_12x12 = 218
    ETC1_RGB8 = 263
    R8 = 280
    R8_SIGNED = 281
    R16 = 282
    R16_SIGNED = 283
    R16F = 284
    R32F = 285
    RG8 = 286
    RG8_SIGNED = 287
    RG16 = 288
    RG16_SIGNED = 289
    RG16F = 290
    RG32F = 291
    RGB8 = 292
    RGB8_SIGNED = 293
    RGB16 = 294
    RGB16_SIGNED = 295
    RGB16F = 296
    RGB32F = 297
    RGBA8 = 298
    RGBA8_SIGNED = 299
    RGBA16 = 300
    RGBA16_SIGNED = 301
    RGBA16F = 302
    RGBA32F = 303
    BGR8 = 304
    BGRA8 = 305
    BGRA8_SRGB = 306
    PVRTC1_RGB2 = 307
    PVRTC1_SRGB2 = 308
    PVRTC1_RGB4 = 309
    PVRTC1_SRGB4 = 310
    PVRTC1_RGBA2 = 311
    PVRTC1_SRGBA2 = 312
    PVRTC1_RGBA4 = 313
    PVRTC1_SRGBA4 = 314
    PVRTC2_RGBA2 = 315
    PVRTC2_SRGBA2 = 316
    PVRTC2_RGBA4 = 317
    PVRTC2_SRGBA4 = 318
    RGBA8Unorm = 319
    RGBA8Unorm_sRGB = 320
    RGB8Unorm = 321
    RGB8Unorm_sRGB = 322
    RG8Unorm = 323
    R8Unorm = 324
    BGRA8Unorm = 325
    BGR8Unorm = 326
    RGBA8Unorm_70 = 70


class Texture:
    def __init__(self, Pixel: ScPixel, Width: int = 0, Height: int = 0) -> None:
        self.Width = Width
        self.Height = Height
        self.DataLength: int = 0
        self.Data: bytes = None
        self.PixelType: ScPixel = Pixel
        self.IsCompressedDataFlag = None
        self.DecompressedData = None
        
    def GetFormatName(self):
        if isinstance(self.PixelType, ScPixel):
            if self.PixelType == ScPixel.RGBA8Unorm_70:
                return "RGBA8Unorm"
            return self.PixelType.name
        else:
            AstcFormatIds = list(range(186, 201)) + list(range(204, 219))
            if self.PixelType in AstcFormatIds:
                AstcIdToName = {
                    186: "ASTC_SRGBA8_4x4", 187: "ASTC_SRGBA8_5x4", 188: "ASTC_SRGBA8_5x5",
                    189: "ASTC_SRGBA8_6x5", 190: "ASTC_SRGBA8_6x6", 192: "ASTC_SRGBA8_8x5",
                    193: "ASTC_SRGBA8_8x6", 194: "ASTC_SRGBA8_8x8", 195: "ASTC_SRGBA8_10x5",
                    196: "ASTC_SRGBA8_10x6", 197: "ASTC_SRGBA8_10x8", 198: "ASTC_SRGBA8_10x10",
                    199: "ASTC_SRGBA8_12x10", 200: "ASTC_SRGBA8_12x12", 204: "ASTC_RGBA8_4x4",
                    205: "ASTC_RGBA8_5x4", 206: "ASTC_RGBA8_5x5", 207: "ASTC_RGBA8_6x5",
                    208: "ASTC_RGBA8_6x6", 210: "ASTC_RGBA8_8x5", 211: "ASTC_RGBA8_8x6",
                    212: "ASTC_RGBA8_8x8", 213: "ASTC_RGBA8_10x5", 214: "ASTC_RGBA8_10x6",
                    215: "ASTC_RGBA8_10x8", 216: "ASTC_RGBA8_10x10", 217: "ASTC_RGBA8_12x10",
                    218: "ASTC_RGBA8_12x12"
                }
                return AstcIdToName.get(self.PixelType, f"ASTC_UNKNOWN_{self.PixelType}")
            
            if self.PixelType == 70:
                return "RGBA8Unorm"
            return f"UNKNOWN ({self.PixelType})"
            
    def IsAstc(self):
        FormatName = self.GetFormatName()
        if "ASTC" in FormatName:
            return True
        
        AstcFormatIds = list(range(186, 201)) + list(range(204, 219))
        if hasattr(self.PixelType, 'value'):
            return self.PixelType.value in AstcFormatIds
        else:
            return self.PixelType in AstcFormatIds
        
    def IsCompressedData(self):
        if self.IsCompressedDataFlag is not None:
            return self.IsCompressedDataFlag
            
        if not self.Data or len(self.Data) < 4:
            self.IsCompressedDataFlag = False
            return False
        
        if self.Data[:4] == b'\x28\xb5\x2f\xfd':
            self.IsCompressedDataFlag = True
            return True
        
        try:
            Dctx = zstandard.ZstdDecompressor()
            TestData = self.Data[:min(100, len(self.Data))]
            Dctx.decompress(TestData, max_output_size=1000)
            self.IsCompressedDataFlag = True
            return True
        except:
            pass
        
        self.IsCompressedDataFlag = False
        return False
        
    def DecompressData(self):
        if self.DecompressedData is not None:
            return self.DecompressedData
            
        if not self.IsCompressedData():
            self.DecompressedData = self.Data
            return self.Data
            
        try:
            Dctx = zstandard.ZstdDecompressor()
            MaxExpected = self.CalculateExpectedSize() * 3
            if MaxExpected == 0:
                MaxExpected = 100 * 1024 * 1024
                
            self.DecompressedData = Dctx.decompress(self.Data, max_output_size=MaxExpected)
            logging.info(f"Successfully decompressed: {len(self.Data)} -> {len(self.DecompressedData)} bytes")
            return self.DecompressedData
        except Exception as E:
            self.DecompressedData = self.Data
            return self.Data
        
    def IsEtc(self):
        FormatName = self.GetFormatName()
        return "ETC" in FormatName or "EAC" in FormatName
        
    def IsSrgb(self):
        FormatName = self.GetFormatName()
        return "SRGB" in FormatName or "SRGBA" in FormatName or "sRGB" in FormatName
        
    def IsPvrtc(self):
        FormatName = self.GetFormatName()
        return "PVRTC" in FormatName
        
    def IsUncompressed(self):
        FormatName = self.GetFormatName()
        UncompressedFormats = ["R8", "R16", "R16F", "R32F", "RG8", "RG16", "RG16F", "RG32F", 
                                "RGB8", "RGB16", "RGB16F", "RGB32F", "RGBA8", "RGBA16", "RGBA16F", 
                                "RGBA32F", "BGR8", "BGRA8", "RGBA8Unorm", "RGB8Unorm", "RG8Unorm", 
                                "R8Unorm", "BGRA8Unorm", "BGR8Unorm"]
        
        if self.IsAstc() or self.IsEtc() or self.IsPvrtc():
            return False
            
        return any(Fmt in FormatName for Fmt in UncompressedFormats)
    
    def CalculateExpectedSize(self):
        if self.Width == 0 or self.Height == 0:
            return 0
            
        FormatName = self.GetFormatName()
        
        if self.IsAstc():
            AstcBlockSizes = {
                "4x4": (4, 4), "5x4": (5, 4), "5x5": (5, 5), "6x5": (6, 5),
                "6x6": (6, 6), "8x5": (8, 5), "8x6": (8, 6), "8x8": (8, 8),
                "10x5": (10, 5), "10x6": (10, 6), "10x8": (10, 8), "10x10": (10, 10),
                "12x10": (12, 10), "12x12": (12, 12)
            }
            
            for BlockPattern, (Bw, Bh) in AstcBlockSizes.items():
                if BlockPattern in FormatName:
                    BlocksX = (self.Width + Bw - 1) // Bw
                    BlocksY = (self.Height + Bh - 1) // Bh
                    return BlocksX * BlocksY * 16
        
        elif self.IsEtc():
            BlocksX = (self.Width + 3) // 4
            BlocksY = (self.Height + 3) // 4
            if "RGBA" in FormatName:
                return BlocksX * BlocksY * 16
            else:
                return BlocksX * BlocksY * 8
        
        elif self.IsUncompressed():
            if "RGBA" in FormatName or "BGRA" in FormatName:
                return self.Width * self.Height * 4
            elif "RGB" in FormatName or "BGR" in FormatName:
                return self.Width * self.Height * 3
            elif "RG" in FormatName:
                return self.Width * self.Height * 2
            elif "R" in FormatName:
                return self.Width * self.Height
        
        return 0


class SCTX:
    def __init__(self, Filepath: str, StreamingIdOverride: int = 0xFF) -> None:
        self.Width = 0
        self.Height = 0
        self.StreamingTextureId = StreamingIdOverride 
        self.TextureId = 0xFF
        self.StreamingTexture: Texture = None
        self.Texture: Texture = None
        self.OriginalFileData = None
        self.CompressedPayload = None
        self.DecompressedPayload = None
        
        FileSize = os.path.getsize(Filepath)
        
        with open(Filepath, "rb") as F:
            self.OriginalFileData = F.read()
        
        self.FindAndDecompressPayload()
        
        Reader = BinaryReader(self.OriginalFileData)
        
        StreamingLength = Reader.ReadUint()
        if StreamingLength > len(self.OriginalFileData) - Reader.tell():
            logging.warning(f"Invalid streaming length {StreamingLength}")
            StreamingLength = min(StreamingLength, len(self.OriginalFileData) - Reader.tell())
        
        StreamingData = Reader.read(StreamingLength)
        self.ReadStreamingData(StreamingData)
        
        if Reader.tell() < len(self.OriginalFileData) - 4:
            DataLength = Reader.ReadUint()
            if DataLength > len(self.OriginalFileData) - Reader.tell():
                logging.warning(f"Invalid data length {DataLength}")
                DataLength = min(DataLength, len(self.OriginalFileData) - Reader.tell())
            
            Data = Reader.read(DataLength)
            self.ReadTexture(Data)
            
            if self.Texture and self.Texture.DataLength > 0:
                if Reader.tell() + self.Texture.DataLength <= len(self.OriginalFileData):
                    self.Texture.Data = Reader.read(self.Texture.DataLength)
                else:
                    logging.warning(f"Texture data length {self.Texture.DataLength} exceeds file size")
                    Remaining = len(self.OriginalFileData) - Reader.tell()
                    if Remaining > 0:
                        self.Texture.Data = Reader.read(Remaining)
        
        if self.Texture and (not self.Texture.Data or len(self.Texture.Data) < 16):
            logging.info("Main texture data is too small, checking streaming texture...")
            if self.StreamingTexture and self.StreamingTexture.Data:
                logging.info("Using streaming texture data instead")
                self.Texture.Data = self.StreamingTexture.Data
                self.Texture.DataLength = self.StreamingTexture.DataLength

    def FindAndDecompressPayload(self):
        ZstdMagic = b'\x28\xb5\x2f\xfd'
        
        for Offset in [0, 512, 536, 584, 768, 1024]:
            if Offset < len(self.OriginalFileData) - 4:
                Chunk = self.OriginalFileData[Offset:Offset+4]
                if Chunk == ZstdMagic:
                    logging.info(f"Found Zstd compressed data at offset {Offset}")
                    self.CompressedPayload = self.OriginalFileData[Offset:]
                    try:
                        Dctx = zstandard.ZstdDecompressor()
                        self.DecompressedPayload = Dctx.decompress(self.CompressedPayload)
                        logging.info(f"Decompressed payload: {len(self.CompressedPayload)} -> {len(self.DecompressedPayload)} bytes")
                        return
                    except Exception as E:
                        logging.error(f"Failed to decompress payload: {E}")
        
        Pos = self.OriginalFileData.find(ZstdMagic)
        if Pos != -1:
            logging.info(f"Found Zstd compressed data at offset {Pos}")
            self.CompressedPayload = self.OriginalFileData[Pos:]
            try:
                Dctx = zstandard.ZstdDecompressor()
                self.DecompressedPayload = Dctx.decompress(self.CompressedPayload)
                logging.info(f"Decompressed payload: {len(self.CompressedPayload)} -> {len(self.DecompressedPayload)} bytes")
            except Exception as E:
                logging.error(f"Failed to decompress payload: {E}")
    
    def ReadStreamingData(self, Data: bytes):
        if len(Data) < 4:
            return
            
        Reader = BinaryReader(Data)
        HeaderLength = Reader.ReadUint()
        if HeaderLength > len(Data) - 4:
            return
            
        Reader.Skip(HeaderLength)
        
        if len(Data) - Reader.tell() < 14:
            return
            
        PixelType = Reader.ReadUint()
        Width = Reader.ReadUshort()
        Height = Reader.ReadUshort()
        Reader.ReadInt()

        AstcFormatIds = list(range(186, 201)) + list(range(204, 219))
        if PixelType in AstcFormatIds:
            logging.info(f"Detected ASTC format: ID {PixelType}")
            try:
                self.Texture = Texture(ScPixel(PixelType), Width, Height)
            except ValueError:
                self.Texture = Texture(PixelType, Width, Height)
        elif PixelType == 70:
            self.Texture = Texture(ScPixel.RGBA8Unorm_70, Width, Height)
        else:
            try:
                self.Texture = Texture(ScPixel(PixelType), Width, Height)
            except ValueError:
                logging.warning(f"Unknown pixel format detected: {PixelType}")
                self.Texture = Texture(PixelType, Width, Height)
                
        if len(Data) - Reader.tell() >= 4:
            self.Texture.DataLength = Reader.ReadUint()
                
        if (self.StreamingTextureId != 0 and len(Data) - Reader.tell() >= 20):
            Reader.Skip(16)
            
            if len(Data) - Reader.tell() >= 4:
                StreamingTextureLength = Reader.ReadUint()
                if StreamingTextureLength > 0 and len(Data) - Reader.tell() >= StreamingTextureLength:
                    self.ReadStreamingTexture(Reader.read(StreamingTextureLength))
                    
            if len(Data) - Reader.tell() >= 4:
                self.StreamingId = Reader.ReadUint()
        
    def ReadStreamingTexture(self, Data: bytes):
        if len(Data) < 28 + 12:
            return
            
        Reader = BinaryReader(Data)
        Reader.Skip(28)
        
        Width = Reader.ReadUshort()
        Height = Reader.ReadUshort()
        PixelType = Reader.ReadUint()
        Reader.ReadInt()

        AstcFormatIds = list(range(186, 201)) + list(range(204, 219))
        if PixelType in AstcFormatIds:
            try:
                self.StreamingTexture = Texture(ScPixel(PixelType), Width, Height)
            except ValueError:
                self.StreamingTexture = Texture(PixelType, Width, Height)
        elif PixelType == 70:
            self.StreamingTexture = Texture(ScPixel.RGBA8Unorm_70, Width, Height)
        else:
            try:
                self.StreamingTexture = Texture(ScPixel(PixelType), Width, Height)
            except ValueError:
                self.StreamingTexture = Texture(PixelType, Width, Height)
            
        if len(Data) - Reader.tell() >= 4:
            self.StreamingTexture.DataLength = Reader.ReadUint()
            if self.StreamingTexture.DataLength > 0 and len(Data) - Reader.tell() >= self.StreamingTexture.DataLength:
                self.StreamingTexture.Data = Reader.read(self.StreamingTexture.DataLength)
        
    def ReadTexture(self, Data: bytes):
        if len(Data) < 24 + 10:
            return
            
        Reader = BinaryReader(Data)
        Reader.Skip(24)
        
        self.Texture.Width = Reader.ReadUshort()
        self.Texture.Height = Reader.ReadUshort()
        Reader.ReadUint()
        
        if len(Data) - Reader.tell() >= 4:
            HashLength = Reader.ReadUint()
            if HashLength > 0 and len(Data) - Reader.tell() >= HashLength:
                HashData = Reader.read(HashLength)

    def LogInfo(self):
        if self.Texture:
            ExpectedSize = self.Texture.CalculateExpectedSize()
            print(f"\nMain Texture:")
            logging.info(f"   Dimensions: {self.Texture.Width} x {self.Texture.Height}")
            logging.info(f"   Pixel Format: {self.Texture.GetFormatName()}")
            logging.info(f"   Format ID: {self.Texture.PixelType}")
            logging.info(f"   Data Length: {self.Texture.DataLength} bytes")
            logging.info(f"   Expected Size: {ExpectedSize} bytes")
            logging.info(f"   Data Compressed: {self.Texture.IsCompressedData()}")
            
            if self.Texture.Data:
                logging.info(f"   Actual Data Size: {len(self.Texture.Data)} bytes")
        
        if self.StreamingTexture:
            ExpectedSize = self.StreamingTexture.CalculateExpectedSize()
            print(f"\nStreaming Texture:")
            logging.info(f"   Dimensions: {self.StreamingTexture.Width} x {self.StreamingTexture.Height}")
            logging.info(f"   Pixel Format: {self.StreamingTexture.GetFormatName()}")
            logging.info(f"   Data Length: {self.StreamingTexture.DataLength} bytes")
            logging.info(f"   Expected Size: {ExpectedSize} bytes")

    def DecodeTexture(self, TextureObj: Texture, UseDecompressedPayload: bool = False):
        if UseDecompressedPayload and self.DecompressedPayload:
            logging.info("Using decompressed payload for decoding")
            TextureData = self.DecompressedPayload
        elif TextureObj.Data:
            TextureData = TextureObj.Data
            if TextureObj.IsCompressedData():
                TextureData = TextureObj.DecompressData()
        else:
            logging.error("No texture data available")
            return None, None
            
        if not TextureData or len(TextureData) < 16:
            logging.error(f"Texture data too small: {len(TextureData) if TextureData else 0} bytes")
            return None, None
        
        Width = TextureObj.Width
        Height = TextureObj.Height
        FormatName = TextureObj.GetFormatName()
        
        try:
            if TextureObj.IsAstc():
                import re
                Match = re.search(r'(\d+)x(\d+)', FormatName)
                if Match:
                    BlockWidth = int(Match.group(1))
                    BlockHeight = int(Match.group(2))
                    RgbaData = texture2ddecoder.decode_astc(TextureData, Width, Height, BlockWidth, BlockHeight)
                    BgraData = bytearray()
                    for I in range(0, len(RgbaData), 4):
                        R, G, B, A = RgbaData[I:I+4]
                        BgraData.extend([B, G, R, A])
                    return bytes(BgraData), 'RGBA'
    
            elif TextureObj.IsEtc():
                if "ETC1" in FormatName or "ETC2_RGB8" in FormatName or "ETC2_SRGB8" in FormatName:
                    RgbaData = texture2ddecoder.decode_etc1(TextureData, Width, Height)
                    BgraData = bytearray()
                    for I in range(0, len(RgbaData), 4):
                        R, G, B, A = RgbaData[I:I+4]
                        BgraData.extend([B, G, R, A])
                    return bytes(BgraData), 'RGBA'
                elif "ETC2_EAC_RGBA8" in FormatName or "ETC2_EAC_SRGBA8" in FormatName:
                    RgbaData = texture2ddecoder.decode_etc2(TextureData, Width, Height)
                    BgraData = bytearray()
                    for I in range(0, len(RgbaData), 4):
                        R, G, B, A = RgbaData[I:I+4]
                        BgraData.extend([B, G, R, A])
                    return bytes(BgraData), 'RGBA'
                elif "ETC2_RGB8_PUNCHTHROUGH_ALPHA1" in FormatName or "ETC2_SRGB8_PUNCHTHROUGH_ALPHA1" in FormatName:
                    RgbaData = texture2ddecoder.decode_etc2a1(TextureData, Width, Height)
                    BgraData = bytearray()
                    for I in range(0, len(RgbaData), 4):
                        R, G, B, A = RgbaData[I:I+4]
                        BgraData.extend([B, G, R, A])
                    return bytes(BgraData), 'RGBA'
                elif "EAC_R11" in FormatName or "EAC_SIGNED_R11" in FormatName:
                    MonoData = texture2ddecoder.decode_eacr(TextureData, Width, Height, signed=("SIGNED" in FormatName))
                    BgraData = bytearray()
                    for Intensity in MonoData:
                        BgraData.extend([Intensity, Intensity, Intensity, 255])
                    return bytes(BgraData), 'RGBA'
                elif "EAC_RG11" in FormatName or "EAC_SIGNED_RG11" in FormatName:
                    RgData = texture2ddecoder.decode_eacrg(TextureData, Width, Height, signed=("SIGNED" in FormatName))
                    BgraData = bytearray()
                    for I in range(0, len(RgData), 2):
                        R, G = RgData[I:I+2]
                        BgraData.extend([0, G, R, 255])
                    return bytes(BgraData), 'RGBA'
    
            elif TextureObj.IsUncompressed():
               
                if "RGBA8Unorm" in FormatName:
                    logging.info("RGBA8Unorm: treating as BGRA (no conversion)")
                    
                    return TextureData, 'RGBA'
                elif "RGBA8" in FormatName and "Unorm" not in FormatName:
                    
                    BgraData = bytearray()
                    for I in range(0, len(TextureData), 4):
                        R, G, B, A = TextureData[I:I+4]
                        BgraData.extend([B, G, R, A])
                    return bytes(BgraData), 'RGBA'
                elif "BGRA8" in FormatName or "BGRA8Unorm" in FormatName or "BGRA8_SRGB" in FormatName:
                    
                    return TextureData, 'RGBA'
                elif "RGB8" in FormatName or "RGB8Unorm" in FormatName or "RGB8_SRGB" in FormatName or "RGB8Unorm_sRGB" in FormatName:
                    RgbData = bytearray()
                    for I in range(0, len(TextureData), 3):
                        R, G, B = TextureData[I:I+3]
                        RgbData.extend([R, G, B])
                    return bytes(RgbData), 'RGB'
                elif "BGR8" in FormatName or "BGR8Unorm" in FormatName:
                    RgbData = bytearray()
                    for I in range(0, len(TextureData), 3):
                        B, G, R = TextureData[I:I+3]
                        RgbData.extend([R, G, B])
                    return bytes(RgbData), 'RGB'
                elif "RG8" in FormatName or "RG8Unorm" in FormatName:
                    RgbData = bytearray()
                    for I in range(0, len(TextureData), 2):
                        R, G = TextureData[I:I+2]
                        RgbData.extend([R, G, 0])
                    return bytes(RgbData), 'RGB'
                elif "R8" in FormatName or "R8Unorm" in FormatName:
                    RgbData = bytearray()
                    for R in TextureData:
                        RgbData.extend([R, R, R])
                    return bytes(RgbData), 'RGB'
                elif "R16F" in FormatName or "R32F" in FormatName:
                    RgbData = bytearray()
                    for I in range(0, len(TextureData), 2 if "R16F" in FormatName else 4):
                        R = TextureData[I]
                        RgbData.extend([R, R, R])
                    return bytes(RgbData), 'RGB'
                elif "RG16F" in FormatName or "RG32F" in FormatName:
                    RgbData = bytearray()
                    Step = 4 if "RG16F" in FormatName else 8
                    for I in range(0, len(TextureData), Step):
                        R = TextureData[I]
                        G = TextureData[I + (2 if "RG16F" in FormatName else 4)]
                        RgbData.extend([R, G, 0])
                    return bytes(RgbData), 'RGB'
                elif "RGB16F" in FormatName or "RGB32F" in FormatName:
                    RgbData = bytearray()
                    Step = 6 if "RGB16F" in FormatName else 12
                    for I in range(0, len(TextureData), Step):
                        R = TextureData[I]
                        G = TextureData[I + (2 if "RGB16F" in FormatName else 4)]
                        B = TextureData[I + (4 if "RGB16F" in FormatName else 8)]
                        RgbData.extend([R, G, B])
                    return bytes(RgbData), 'RGB'
                elif "RGBA16F" in FormatName or "RGBA32F" in FormatName:
                    BgraData = bytearray()
                    Step = 8 if "RGBA16F" in FormatName else 16
                    for I in range(0, len(TextureData), Step):
                        R = TextureData[I]
                        G = TextureData[I + (2 if "RGBA16F" in FormatName else 4)]
                        B = TextureData[I + (4 if "RGBA16F" in FormatName else 8)]
                        A = TextureData[I + (6 if "RGBA16F" in FormatName else 12)]
                        BgraData.extend([B, G, R, A])
                    return bytes(BgraData), 'RGBA'
    
            elif TextureObj.IsPvrtc():
                Is2Bpp = "2" in FormatName
                RgbaData = texture2ddecoder.decode_pvrtc(TextureData, Width, Height, Is2Bpp)
                BgraData = bytearray()
                for I in range(0, len(RgbaData), 4):
                    R, G, B, A = RgbaData[I:I+4]
                    BgraData.extend([B, G, R, A])
                return bytes(BgraData), 'RGBA'
    
            logging.error(f"Unsupported texture format: {FormatName}")
            return None, None
                
        except Exception as E:
            logging.error(f"Error decoding texture: {E}")
            import traceback
            traceback.print_exc()
            return None, None


def GenerateOutputFilename(InputPath):
    BaseName = os.path.splitext(os.path.basename(InputPath))[0]
    return f"{BaseName}.png"


def ProcessSingleFile(Args):
   
    InputFile, OutputFile = Args
    try:
        Ctx = SCTX(InputFile)
        
        TextureToDecode = Ctx.Texture
        if not TextureToDecode and Ctx.StreamingTexture:
            TextureToDecode = Ctx.StreamingTexture
            logging.info(f"[{InputFile}] Using streaming texture for decoding")
        
        if not TextureToDecode:
            logging.error(f"[{InputFile}] No texture found in SCTX file")
            return False, InputFile, "No texture found"
        
        ImageData, ImageMode = Ctx.DecodeTexture(TextureToDecode, UseDecompressedPayload=False)

        if not ImageData and Ctx.DecompressedPayload:
            logging.info(f"[{InputFile}] Retrying with decompressed payload...")
            ImageData, ImageMode = Ctx.DecodeTexture(TextureToDecode, UseDecompressedPayload=True)
        
        if ImageData:
            ImageObj = Image.frombytes(ImageMode, (TextureToDecode.Width, TextureToDecode.Height), ImageData)
            ImageObj.save(OutputFile)
            return True, InputFile, OutputFile
        else:
            return False, InputFile, "Failed to decode texture"
            
    except Exception as E:
        return False, InputFile, str(E)


def ProcessBatchFiles(InputFiles, OutputDir=None):
    
    NumCores = cpu_count()
    logging.info(f"Using {NumCores} CPU cores for parallel processing")
    
    Tasks = []
    for InputFile in InputFiles:
        if OutputDir:
            os.makedirs(OutputDir, exist_ok=True)
            BaseName = os.path.splitext(os.path.basename(InputFile))[0]
            OutputFile = os.path.join(OutputDir, f"{BaseName}.png")
        else:
            OutputFile = GenerateOutputFilename(InputFile)
        Tasks.append((InputFile, OutputFile))
    
    SuccessCount = 0
    FailCount = 0
    
    with Pool(processes=NumCores) as Pool_:
        Results = Pool_.map(ProcessSingleFile, Tasks)
        
        for Success, InputFile, Message in Results:
            if Success:
                print(f"✓ {InputFile} -> {Message}")
                SuccessCount += 1
            else:
                print(f"✗ {InputFile}: {Message}")
                FailCount += 1
    
    print(f"\nProcessing complete: {SuccessCount} succeeded, {FailCount} failed")
    return SuccessCount, FailCount


if __name__ == "__main__":
    if len(sys.argv) < 2:
       
        print("Single file: python SctxDecode.py <Input.sctx> [Output.png]")
        print("Batch mode:  python SctxDecode.py <Input1.sctx> <Input2.sctx> -o <OutputDir>")
        print("Directory:   python SctxDecode.py <InputDir> -o <OutputDir>")
        sys.exit(1)
    
    
    if len(sys.argv) > 3 or (len(sys.argv) == 3 and sys.argv[2] == '-o'):
        
        InputFiles = []
        OutputDir = None
        
        if os.path.isdir(sys.argv[1]):
            
            InputDir = sys.argv[1]
            InputFiles = [os.path.join(InputDir, f) for f in os.listdir(InputDir) 
                         if f.lower().endswith('.sctx')]
            if '-o' in sys.argv:
                OutputDir = sys.argv[sys.argv.index('-o') + 1]
        else:
           
            for Arg in sys.argv[1:]:
                if Arg == '-o':
                    break
                if os.path.exists(Arg):
                    InputFiles.append(Arg)
            
            if '-o' in sys.argv:
                OutputDir = sys.argv[sys.argv.index('-o') + 1]
        
        if not InputFiles:
            logging.error("No valid input files found")
            sys.exit(1)
        
        ProcessBatchFiles(InputFiles, OutputDir)
    
    else:
        
        InputFile = sys.argv[1]
        OutputFile = sys.argv[2] if len(sys.argv) > 2 else GenerateOutputFilename(InputFile)
        
        if not os.path.exists(InputFile):
            logging.error(f"File not found: {InputFile}")
            sys.exit(1)
        
        try:
            Ctx = SCTX(InputFile)
            
            Ctx.LogInfo()
            
            TextureToDecode = Ctx.Texture
            if not TextureToDecode and Ctx.StreamingTexture:
                TextureToDecode = Ctx.StreamingTexture
                logging.info("Using streaming texture for decoding")
            
            if not TextureToDecode:
                logging.error("No texture found in SCTX file")
                sys.exit(1)
            
            ImageData, ImageMode = Ctx.DecodeTexture(TextureToDecode, UseDecompressedPayload=False)

            if not ImageData and Ctx.DecompressedPayload:
                logging.info("Retrying with decompressed payload...")
                ImageData, ImageMode = Ctx.DecodeTexture(TextureToDecode, UseDecompressedPayload=True)
            
            if ImageData:
                ImageObj = Image.frombytes(ImageMode, (TextureToDecode.Width, TextureToDecode.Height), ImageData)
                ImageObj.save(OutputFile)
                print(f"Output: {OutputFile}")
            else:
                logging.error("Failed to decode texture")
                sys.exit(1)
                
        except Exception as E:
            logging.error(f"Error processing file: {E}")
            import traceback
            traceback.print_exc()
            sys.exit(1)