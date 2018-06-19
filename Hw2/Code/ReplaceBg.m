function [output] = ReplaceBg(input, mask, eBg)

mask_eBg = eBg .* (1 - mask);
extract_Face = input .* mask;
output = extract_Face + mask_eBg;

end

